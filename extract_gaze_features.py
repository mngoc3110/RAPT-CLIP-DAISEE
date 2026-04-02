import os
import cv2
import numpy as np
import mediapipe as mp
import glob
from multiprocessing import Pool
import tqdm

# MediaPipe landmark indices
LEFT_EYE = [362, 385, 386, 263, 374, 380]
RIGHT_EYE = [33, 159, 158, 133, 153, 145]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# 3D model points for pose estimation
face3d = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def calculate_ear(eye_landmarks):
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def process_video(video_path, output_dir):
    # Extract just the video ID (e.g. 1100011002)
    vid_id = os.path.basename(video_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{vid_id}.npy")
    
    # If already exists, skip
    if os.path.exists(save_path):
        return True
        
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    features = [] # List of [avg_ear, pitch, yaw] per frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            pts = np.array([np.array([l.x * w, l.y * h]) for l in landmarks])
            
            # EAR
            left_ear = calculate_ear(pts[LEFT_EYE])
            right_ear = calculate_ear(pts[RIGHT_EYE])
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Pose Estimation (Pitch, Yaw)
            image_pts = np.array([
                tuple(pts[NOSE_TIP]),
                tuple(pts[CHIN]),
                tuple(pts[LEFT_EYE_CORNER]),
                tuple(pts[RIGHT_EYE_CORNER]),
                tuple(pts[LEFT_MOUTH_CORNER]),
                tuple(pts[RIGHT_MOUTH_CORNER])
            ], dtype=np.float64)
            
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype=np.float64
            )
            dist_coeffs = np.zeros((4, 1))
            
            success, rotation_vector, translation_vector = cv2.solvePnP(
                face3d, image_pts, camera_matrix, dist_coeffs)
                
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                proj_matrix = np.hstack((rmat, translation_vector))
                euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
                pitch = euler_angles[0][0]
                yaw = euler_angles[1][0]
            else:
                pitch, yaw = 0.0, 0.0
                
            features.append([avg_ear, pitch, yaw])
        else:
            # Face not detected, pad with zeros or previous frame
            if len(features) > 0:
                features.append(features[-1]) # copy previous
            else:
                features.append([0.0, 0.0, 0.0])
                
    cap.release()
    face_mesh.close()
    
    # Save as numpy array
    np.save(save_path, np.array(features, dtype=np.float32))
    return True

if __name__ == '__main__':
    # Kaggle dataset path
    root_dir = "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
    if not os.path.exists(root_dir):
        root_dir = "/Users/macbook/Downloads/RAPT-CLIP-DAISEE/DAiSEE" # Local fallback
        
    # Sửa: Đường dẫn xuất file. Vì Kaggle /input là read-only, ta phải lưu ra /working
    output_dir = "/kaggle/working/Gaze_Features"
    if not os.path.exists("/kaggle/working"):
        output_dir = os.path.join(root_dir, "Gaze_Features") # Local fallback
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Features will be saved to: {output_dir}")
        
    all_videos = []
    for ext in ["*.avi", "*.mp4"]:
        all_videos.extend(glob.glob(f"{root_dir}/*/*/*/{ext}", recursive=True))
        # Support deep CAER/DAiSEE folder structures
        all_videos.extend(glob.glob(f"{root_dir}/*/*/*/*/{ext}", recursive=True))
        
    print(f"Found {len(all_videos)} videos to process.")
    
    # Bundle args for map
    tasks = [(vid, output_dir) for vid in all_videos]
    
    # Process in parallel
    num_cores = max(1, os.cpu_count() - 1)
    print(f"Using {num_cores} cores for parallel extraction...")
    
    with Pool(num_cores) as p:
        list(tqdm.tqdm(p.starmap(process_video, tasks), total=len(tasks)))
        
    print("FINISHED EXTRACTION!")
