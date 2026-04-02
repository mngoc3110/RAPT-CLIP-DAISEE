import os
import cv2
import numpy as np
import glob
from concurrent.futures import ThreadPoolExecutor
import tqdm
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Download model if not exists
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

# Create FaceLandmarker options with GPU Delegate
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH,
    delegate=python.BaseOptions.Delegate.GPU
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

# Shared across threads
detector = vision.FaceLandmarker.create_from_options(options)

# Landmark Indices for EAR
LEFT_EYE = [362, 385, 386, 263, 374, 380]
RIGHT_EYE = [33, 159, 158, 133, 153, 145]

def calculate_ear(pts, eye_indices):
    eye_pts = pts[eye_indices]
    if len(eye_pts) != 6: return 0.0
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    h = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def process_video_tasks_api(args):
    video_path, output_dir = args
    vid_id = os.path.basename(video_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{vid_id}.npy")
    
    if os.path.exists(save_path):
        return True
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    features = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to MP Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect
        detection_result = detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            # Convert normalized landmarks to pixel coords
            h, w, _ = frame.shape
            pts = np.array([[l.x * w, l.y * h] for l in landmarks])
            
            # EAR
            left_ear = calculate_ear(pts, LEFT_EYE)
            right_ear = calculate_ear(pts, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Pose Estimation (Pitch/Yaw) from facial_transformation_matrixes
            # The Tasks API provides a 4x4 matrix representing the pose!
            if detection_result.facial_transformation_matrixes:
                matrix = detection_result.facial_transformation_matrixes[0]
                # Extract rotation matrix (top-left 3x3)
                r_mat = matrix[:3, :3]
                # Decompose into euler angles
                sy = np.sqrt(r_mat[0,0] * r_mat[0,0] + r_mat[1,0] * r_mat[1,0])
                singular = sy < 1e-6
                if not singular:
                    pitch = np.arctan2(r_mat[2,1], r_mat[2,2])
                    yaw = np.arctan2(-r_mat[2,0], sy)
                else:
                    pitch = np.arctan2(-r_mat[1,2], r_mat[1,1])
                    yaw = np.arctan2(-r_mat[2,0], sy)
                    
                pitch, yaw = np.degrees(pitch), np.degrees(yaw)
            else:
                pitch, yaw = 0.0, 0.0
                
            features.append([avg_ear, float(pitch), float(yaw)])
        else:
            if len(features) > 0:
                features.append(features[-1])
            else:
                features.append([0.0, 0.0, 0.0])
                
    cap.release()
    np.save(save_path, np.array(features, dtype=np.float32))
    return True

if __name__ == '__main__':
    # Kaggle dataset path
    root_dir = "/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
    if not os.path.exists(root_dir):
        root_dir = "/Users/macbook/Downloads/RAPT-CLIP-DAISEE/DAiSEE" # Local fallback
        
    output_dir = "/kaggle/working/Gaze_Features"
    if not os.path.exists("/kaggle/working"):
        output_dir = os.path.join(root_dir, "Gaze_Features") # Local fallback
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Features will be saved to: {output_dir}")
        
    all_videos = []
    for ext in ["*.avi", "*.mp4"]:
        all_videos.extend(glob.glob(f"{root_dir}/*/*/*/{ext}", recursive=True))
        all_videos.extend(glob.glob(f"{root_dir}/*/*/*/*/{ext}", recursive=True))
        
    print(f"Found {len(all_videos)} videos to process.")
    
    tasks = [(vid, output_dir) for vid in all_videos]
    num_cores = max(1, os.cpu_count() - 1)
    print(f"Using {num_cores} threads for parallel extraction...")
    
    with ThreadPoolExecutor(max_workers=num_cores) as pool:
        list(tqdm.tqdm(pool.map(process_video_tasks_api, tasks), total=len(tasks)))
        
    print("FINISHED EXTRACTION VIA TASKS API!")
