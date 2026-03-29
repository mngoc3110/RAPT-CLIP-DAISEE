class_names_5 = [
    "Neutral (student in class).",
    "Enjoyment (student in class).",
    "Confusion (student in class).",
    "Fatigue (student in class).",
    "Distraction (student in class)."
]

class_names_with_context_5 = [
    "A student shows a neutral learning state in a classroom.",
    "A student shows enjoyment while learning in a classroom.",
    "A student shows confusion during learning in a classroom.",
    "A student shows fatigue during learning in a classroom.",
    "A student shows distraction and is not focused in a classroom."
]

class_descriptor_5_only_face = [
    "A student has a neutral face with relaxed mouth, open eyes, and calm eyebrows.",
    "A student looks happy with a slight smile, bright eyes, and relaxed eyebrows.",
    "A student looks confused with furrowed eyebrows, a puzzled look, and slightly open mouth.",
    "A student looks tired with drooping eyelids, frequent yawning, and a sleepy face.",
    "A student looks distracted with unfocused eyes and a wandering gaze away from the lesson."
]

class_descriptor_5_only_body = [
    "A student sits still with an upright posture and hands on the desk, showing a neutral learning state.",
    "A student leans slightly forward with an open, engaged posture, showing enjoyment in learning.",
    "A student tilts the head and leans in, hand on chin, showing confusion while trying to understand.",
    "A student slouches with shoulders dropped and head lowered, showing fatigue during class.",
    "A student shifts around, turns away from the desk, or looks sideways, showing distraction and low focus."
]

class_descriptor_5 = [
    "A student looks neutral and calm in class, with a relaxed face and steady gaze, quietly watching the lecture or reading notes.",
    "A student shows enjoyment while learning, with a gentle smile and bright eyes, appearing engaged and interested in the lesson.",
    "A student looks confused in class, with furrowed eyebrows and a puzzled expression, focusing on the material as if trying to understand.",
    "A student appears fatigued in class, with drooping eyelids and yawning, head slightly lowered, showing low energy.",
    "A student is distracted in class, frequently looking away from the lesson, scanning around, and not paying attention to learning materials."
]

# Prompt Ensemble for RAER (5 classes)
# Each inner list contains multiple descriptions for a single class.
prompt_ensemble_5 = [
    [   # Neutral
        "A photo of a student being alert and looking straight ahead.",
        "A photo of a student with a calm and steady gaze.",
        "A photo of a student paying attention with a neutral expression."
    ],
    [   # Enjoyment
        "A photo of a student smiling and looking happy.",
        "A photo of a student showing joy and enthusiasm.",
        "A photo of a student appearing pleased and engaged."
    ],
    [   # Confusion
        "A photo of a student frowning with a puzzled expression.",
        "A photo of a student scratching their head or looking confused.",
        "A photo of a student trying hard to understand but failing."
    ],
    [   # Fatigue
        "A photo of a student yawning or falling asleep.",
        "A photo of a student with heavy drooping eyelids.",
        "A photo of a student resting their head, looking very tired."
    ],
    [   # Distraction
        "A photo of a student looking away from the screen.",
        "A photo of a student turning their head to the side.",
        "A photo of a student engaging in other activities, not studying."
    ]
]

class_descriptor_8 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.',
    'A person who is feeling contempt.'
]

class_names_8 = [
    'Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'
]

class_names_7 = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

class_descriptor_7 = [
    'A person who is feeling neutral.',
    'A person who is feeling happy.',
    'A person who is feeling sad.',
    'A person who is feeling surprise.',
    'A person who is feeling fear.',
    'A person who is feeling disgust.',
    'A person who is feeling anger.'
]

class_names_with_context_7 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.'
]

class_descriptor_7_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.'
]

class_descriptor_7_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.'
]

class_names_with_context_8 = [
    'A person shows neutral emotion.',
    'A person shows happy emotion.',
    'A person shows sad emotion.',
    'A person shows surprise emotion.',
    'A person shows fear emotion.',
    'A person shows disgust emotion.',
    'A person shows anger emotion.',
    'A person shows contempt emotion.'
]

class_descriptor_8_only_face = [
    'The face of a person who is feeling neutral.',
    'The face of a person who is feeling happy.',
    'The face of a person who is feeling sad.',
    'The face of a person who is feeling surprise.',
    'The face of a person who is feeling fear.',
    'The face of a person who is feeling disgust.',
    'The face of a person who is feeling anger.',
    'The face of a person who is feeling contempt.'
]

class_descriptor_8_only_body = [
    'The body of a person who is feeling neutral.',
    'The body of a person who is feeling happy.',
    'The body of a person who is feeling sad.',
    'The body of a person who is feeling surprise.',
    'The body of a person who is feeling fear.',
    'The body of a person who is feeling disgust.',
    'The body of a person who is feeling anger.',
    'The body of a person who is feeling contempt.'
]

# CK+ Classes (Alphabetical Order: Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise)
class_names_ckplus = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

class_names_with_context_ckplus = [
    "A person shows anger.",
    "A person shows contempt.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_ckplus = [
    "A person with an angry expression, furrowed brows and tightened lips.",
    "A person with a contemptuous expression, one corner of the lip raised.",
    "A person with a disgusted expression, nose wrinkled and upper lip raised.",
    "A person with a fearful expression, eyes wide open and eyebrows raised.",
    "A person with a happy expression, smiling with cheeks raised.",
    "A person with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A person with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_ckplus = [
    [ # Anger
        "A photo of a person showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Contempt
        "A photo of a person showing contempt.",
        "A face with a smirk or sneer.",
        "A contemptuous facial expression."
    ],
    [ # Disgust
        "A photo of a person showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a person showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a person showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Sadness
        "A photo of a person showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a person showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# SFER Classes (Alphabetical: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_sfer = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_sfer = [
    "A student shows anger.",
    "A student shows disgust.",
    "A student shows fear.",
    "A student shows happiness.",
    "A student shows neutrality.",
    "A student shows sadness.",
    "A student shows surprise."
]

class_descriptor_sfer = [
    "A student with an angry expression, furrowed brows and tightened lips.",
    "A student with a disgusted expression, nose wrinkled and upper lip raised.",
    "A student with a fearful expression, eyes wide open and eyebrows raised.",
    "A student with a happy expression, smiling with cheeks raised.",
    "A student with a neutral expression, relaxed face and calm gaze.",
    "A student with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A student with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_sfer = [
    [ # Anger
        "A photo of a student showing anger.",
        "A face with furrowed brows and a glare.",
        "An angry facial expression."
    ],
    [ # Disgust
        "A photo of a student showing disgust.",
        "A face with a wrinkled nose.",
        "A disgusted facial expression."
    ],
    [ # Fear
        "A photo of a student showing fear.",
        "A face with wide eyes and a terrified look.",
        "A fearful facial expression."
    ],
    [ # Happy
        "A photo of a student showing happiness.",
        "A smiling face with joy.",
        "A happy facial expression."
    ],
    [ # Neutral
        "A photo of a student showing a neutral expression.",
        "A calm face with no strong emotion.",
        "A neutral facial expression."
    ],
    [ # Sad
        "A photo of a student showing sadness.",
        "A face with a frown and sorrowful eyes.",
        "A sad facial expression."
    ],
    [ # Surprise
        "A photo of a student showing surprise.",
        "A face with an open mouth and wide eyes.",
        "A surprised facial expression."
    ]
]

# CAER Classes (Alphabetical: Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise)
class_names_caer = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

class_names_with_context_caer = [
    "A person shows anger.",
    "A person shows disgust.",
    "A person shows fear.",
    "A person shows happiness.",
    "A person shows neutrality.",
    "A person shows sadness.",
    "A person shows surprise."
]

class_descriptor_caer = [
    "A person with an angry expression, furrowed brows and tightened lips.",
    "A person with a disgusted expression, nose wrinkled and upper lip raised.",
    "A person with a fearful expression, eyes wide open and eyebrows raised.",
    "A person with a happy expression, smiling with cheeks raised.",
    "A person with a neutral expression, relaxed face and calm gaze.",
    "A person with a sad expression, corners of the lips turned down and drooping eyelids.",
    "A person with a surprised expression, mouth open and eyes widened."
]

prompt_ensemble_caer = [
    [ # Anger
        "A scene from a movie showing a person in an angry confrontation.",
        "A person with an angry face in a tense and hostile environment.",
        "A photo of an actor expressing rage during a dramatic cinematic scene."
    ],
    [ # Disgust
        "A person showing a disgusted expression in a repulsive or unpleasant situation.",
        "A movie scene where someone reacts with revulsion to something gross.",
        "A cinematic shot of a face showing strong dislike and disgust."
    ],
    [ # Fear
        "A person showing fear in a dark, mysterious, or threatening environment.",
        "A scene from a suspenseful movie where an actor looks absolutely terrified.",
        "A cinematic photo of a fearful face during a scary and tense moment."
    ],
    [ # Happy
        "A person feeling happy and celebrating in a bright, social movie scene.",
        "A joyful actor in a pleasant environment showing a wide smile.",
        "A scene showing a happy facial expression during a positive and cheerful event."
    ],
    [ # Neutral
        "A person with a neutral expression in a mundane, everyday cinematic scene.",
        "A calm and expressionless actor in a casual movie background.",
        "A photo of someone showing no specific emotion in a normal setting."
    ],
    [ # Sad
        "A person showing sadness and grief in a gloomy or lonely cinematic environment.",
        "A scene from a dramatic movie where an actor looks heartbroken and sorrowful.",
        "A cinematic photo of a sad face in a somber, quiet, and melancholic setting."
    ],
    [ # Surprise
        "A person showing great surprise at an unexpected event in a dynamic scene.",
        "A shocked facial expression during a surprising and dramatic movie moment.",
        "A cinematic shot of an actor looking amazed or startled by something sudden."
    ]
]

class_names_daisee = ['Low', 'High', 'Very High']

class_names_with_context_daisee = [
    "A student shows low engagement during an online class.",
    "A student shows high engagement during an online class.",
    "A student shows very high engagement during an online class."
]

class_descriptor_daisee = [
    "A student is disengaged or distracted: eyes closed, looking away, unfocused stare, yawning, droopy eyelids, or showing no interest in the screen.",
    "A student is attentively watching the screen with a calm and neutral expression, eyes focused on the lesson content.",
    "A student is highly engaged: eyes wide open tracking content, showing concentration, curiosity, and active interest in the lesson."
]

prompt_ensemble_daisee = [
    [ # Low Engagement (0) = merged Very Low + Low
        "A close-up of a student's face with eyes fully closed, appearing to be asleep during class.",
        "A face with heavy drooping eyelids and a slack jaw, dozing off in front of the screen.",
        "A close-up of a student's face with an unfocused bored expression, eyes half-open.",
        "A student's face with eyes wandering around instead of looking at the screen.",
        "A face showing a tired yawning expression with droopy eyes during an online class.",
        "A student with a distracted look, eyes glancing to the side rather than at the camera.",
        "A close-up showing a face with no eye contact, completely zoned out or barely paying attention."
    ],
    [ # High Engagement (1)
        "A close-up of a student's face looking directly at the screen with calm focused eyes.",
        "A student's face with steady eye contact toward the camera and a neutral attentive expression.",
        "A face showing quiet concentration with relaxed features and eyes fixed ahead.",
        "A student with clear open eyes watching the screen attentively with a composed expression.",
        "A close-up of a face with a focused neutral gaze, paying attention to the online lesson.",
        "A student's face with relaxed brow and steady eyes, quietly absorbing the lecture content.",
        "A face showing sustained attention with gentle focused eyes and a calm mouth."
    ],
    [ # Very High Engagement (2)
        "A close-up of a student's face with wide alert eyes showing intense focus and interest.",
        "A student's face with raised eyebrows and bright eyes, deeply engaged in the lesson.",
        "A face showing an animated expression of curiosity with widened eyes and a slight smile.",
        "A student with slightly furrowed brows and intensely concentrated eyes fixed on the screen.",
        "A close-up of an excited face reacting to content with expressive eyes and an open expression.",
        "A student's face showing visible intellectual engagement with alert widened eyes and a nod.",
        "A face with a bright eager expression, eyes sparkling with interest in the lesson material."
    ]
]

# ============================================================================
# DAiSEE 4-Level Engagement (VeryLow / Low / High / VeryHigh)
# Original 4-level scale, NOT merged
# ============================================================================
class_names_daisee_4level = ['Very Low', 'Low', 'High', 'Very High']

class_names_with_context_daisee_4level = [
    "A student shows very low engagement during an online class.",
    "A student shows low engagement during an online class.",
    "A student shows high engagement during an online class.",
    "A student shows very high engagement during an online class."
]

class_descriptor_daisee_4level = [
    "A student is completely disengaged: eyes closed or nearly closed, head drooping, asleep or about to fall asleep during the online class.",
    "A student shows low engagement: distracted look, eyes wandering, unfocused gaze, yawning, or showing minimal interest in the screen.",
    "A student is attentively watching the screen with calm focused eyes, a neutral expression, and sustained attention on the lesson.",
    "A student is highly engaged: wide alert eyes, raised eyebrows, animated expression showing intense curiosity and active interest."
]

prompt_ensemble_daisee_4level = [
    [ # Very Low Engagement (0) — asleep, eyes closed, head down
        "A close-up of a student's face with eyes completely closed, head tilted down, asleep during class.",
        "A face with eyes shut and mouth slightly open, clearly sleeping in front of the webcam.",
        "A student with head resting on hand, eyes fully closed, completely unconscious during the lesson.",
        "A close-up showing a student's face with heavy closed eyelids and no sign of awareness.",
        "A student appearing to be in deep sleep during an online class, face relaxed and motionless.",
        "A face showing no sign of consciousness, eyes closed, jaw slack, completely asleep.",
        "A student with head drooping forward, eyes sealed shut, not responsive to the online lesson at all."
    ],
    [ # Low Engagement (1) — distracted, bored, looking away, yawning
        "A close-up of a student's face with an unfocused bored expression, eyes half-open and glazed.",
        "A student's face with eyes wandering around the room instead of looking at the screen.",
        "A face showing a tired yawning expression with droopy eyelids during an online class.",
        "A student with a distracted look, eyes glancing to the side rather than at the camera.",
        "A close-up showing a face with no eye contact and a listless uninterested expression.",
        "A student looking away from the screen with a blank disengaged stare.",
        "A face with heavy drooping eyelids and a slack jaw, barely paying attention to the lesson."
    ],
    [ # High Engagement (2) — focused, calm, attentive
        "A close-up of a student's face looking directly at the screen with calm focused eyes.",
        "A student's face with steady eye contact toward the camera and a neutral attentive expression.",
        "A face showing quiet concentration with relaxed features and eyes fixed ahead.",
        "A student with clear open eyes watching the screen attentively with a composed expression.",
        "A close-up of a face with a focused neutral gaze, paying attention to the online lesson.",
        "A student's face with relaxed brow and steady eyes, quietly absorbing the lecture content.",
        "A face showing sustained attention with gentle focused eyes and a calm mouth."
    ],
    [ # Very High Engagement (3) — excited, curious, intensely focused
        "A close-up of a student's face with wide alert eyes showing intense focus and interest.",
        "A student's face with raised eyebrows and bright eyes, deeply engaged in the lesson.",
        "A face showing an animated expression of curiosity with widened eyes and a slight smile.",
        "A student with slightly furrowed brows and intensely concentrated eyes fixed on the screen.",
        "A close-up of an excited face reacting to content with expressive eyes and an open expression.",
        "A student's face showing visible intellectual engagement with alert widened eyes and a nod.",
        "A face with a bright eager expression, eyes sparkling with interest in the lesson material."
    ]
]

# ============================================================================
# DAiSEE Binary (Not Engaged vs Engaged)
# Merge: VeryLow(0)+Low(1) → 0 (Not Engaged), High(2)+VeryHigh(3) → 1 (Engaged)
# ============================================================================
class_names_daisee_binary = ['Not Engaged', 'Engaged']

class_names_with_context_daisee_binary = [
    "A student who is not engaged during an online class.",
    "A student who is engaged during an online class."
]

class_descriptor_daisee_binary = [
    "A student is not engaged: eyes closed or drooping, looking away, yawning, distracted, bored, or showing no interest in the lesson.",
    "A student is engaged: eyes focused on screen, attentive expression, actively watching and following the online lesson content."
]

prompt_ensemble_daisee_binary = [
    [ # Not Engaged (0)
        "A close-up of a student's face with eyes closed or nearly closed, appearing to be asleep during class.",
        "A face with heavy drooping eyelids and a slack jaw, dozing off in front of the screen.",
        "A student's face with an unfocused bored expression, eyes half-open and glazed.",
        "A student with eyes wandering around the room instead of looking at the screen.",
        "A face showing a tired yawning expression with droopy eyes during an online class.",
        "A student with a distracted look, eyes glancing to the side rather than at the camera.",
        "A close-up showing a face with no eye contact, completely zoned out or barely paying attention."
    ],
    [ # Engaged (1)
        "A close-up of a student's face looking directly at the screen with focused attentive eyes.",
        "A student's face with steady eye contact toward the camera and a calm concentrated expression.",
        "A face showing quiet concentration with relaxed features and eyes fixed ahead on the lesson.",
        "A student with clear open eyes watching the screen attentively with a composed expression.",
        "A student's face with raised eyebrows and bright eyes, deeply engaged in the lesson.",
        "A face showing an animated expression of curiosity with widened eyes and interest.",
        "A student with alert wide eyes showing intense focus and active intellectual engagement."
    ]
]

# ============================================================================
# Student Engagement Dataset (Binary: Engaged vs Not Engaged)
# ============================================================================
class_names_student_engagement = [
    "Engaged",
    "Not Engaged"
]

class_names_with_context_student_engagement = [
    "An engaged student in an online class.",
    "A not engaged student in an online class."
]

class_descriptor_student_engagement = [
    "A student is engaged: showing concentration, confusion about content, or frustration while actively trying to follow the lesson.",
    "A student is not engaged: looking away from the screen, appearing bored, or drowsy during the online class."
]

prompt_ensemble_student_engagement = [
    [ # Engaged (0) - includes confused, engaged, frustrated (actively processing)
        "A close-up of a student's face looking directly at the screen with focused attentive eyes.",
        "A student's face showing concentration and mental effort while following an online lesson.",
        "A face with furrowed brows and a confused expression, actively trying to understand the content.",
        "A student with an engaged and alert expression, eyes fixed on the screen during online class.",
        "A close-up showing a student's face with a frustrated but attentive look, struggling with the material.",
        "A student with wide open eyes showing curiosity and active participation in the lesson.",
        "A face showing emotional involvement in the class content, whether concentration, confusion, or effort."
    ],
    [ # Not Engaged (1) - includes looking away, bored, drowsy
        "A close-up of a student's face looking away from the screen, not paying attention to the lesson.",
        "A student's face with droopy eyes and a bored expression during an online class.",
        "A face showing complete disinterest with glazed-over eyes and a listless expression.",
        "A student appearing drowsy with heavy eyelids, about to fall asleep during the online lesson.",
        "A close-up of a student looking at something other than the screen, distracted and unfocused.",
        "A student with a yawning expression and tired face, showing no interest in the class.",
        "A face showing signs of sleepiness and boredom, with the chin resting on hand and eyes half-closed."
    ]
]

# ============================================================================
# Student Engagement 6-Class (subclass level)
# Classes: confused(0), engaged(1), frustrated(2), looking_away(3), bored(4), drowsy(5)
# ============================================================================
class_names_student_engagement_6 = [
    "confused", "engaged", "frustrated", "looking away", "bored", "drowsy"
]

class_names_with_context_student_engagement_6 = [
    "A confused student in an online class.",
    "An engaged student in an online class.",
    "A frustrated student in an online class.",
    "A student looking away from the screen in an online class.",
    "A bored student in an online class.",
    "A drowsy student in an online class."
]

class_descriptor_student_engagement_6 = [
    "A student shows confusion: furrowed brows, tilted head, puzzled expression while trying to understand content.",
    "A student is engaged: focused eyes on screen, attentive posture, actively following the lesson.",
    "A student shows frustration: tense expression, clenched jaw, visible stress while struggling with material.",
    "A student is looking away: eyes directed elsewhere, head turned from screen, not paying attention.",
    "A student is bored: blank stare, slouched posture, no interest or emotional response to content.",
    "A student is drowsy: heavy eyelids, yawning, head drooping, struggling to stay awake."
]

prompt_ensemble_student_engagement_6 = [
    [ # confused (0)
        "A student's face with furrowed brows and a puzzled look, trying to understand the lesson.",
        "A close-up of a confused face with squinted eyes and a tilted head during online class.",
        "A student showing visible confusion with raised eyebrows and an uncertain expression.",
        "A face with a perplexed expression, mouth slightly open, struggling to follow the content.",
        "A student with a questioning look, brow creased, mentally processing difficult material."
    ],
    [ # engaged (1)
        "A student with focused eyes looking directly at the screen, fully attentive to the lesson.",
        "A close-up of an alert face with clear eyes and an interested expression during class.",
        "A student showing active engagement with bright eyes and a slight nod of understanding.",
        "A face with steady eye contact toward the camera, calm and concentrated expression.",
        "A student with an attentive posture and focused gaze, absorbed in the lecture content."
    ],
    [ # frustrated (2)
        "A student's face showing visible frustration with clenched jaw and tense expression.",
        "A close-up of a frustrated face with narrowed eyes and a frown during online class.",
        "A student showing stress and annoyance while struggling with the course material.",
        "A face with pressed lips and an irritated look, frustrated by the difficulty of the content.",
        "A student with a strained expression showing emotional distress during the lesson."
    ],
    [ # looking away (3)
        "A student looking away from the screen, eyes directed to the side or at something else.",
        "A close-up of a student with head turned, not facing the camera or screen at all.",
        "A student whose gaze is directed elsewhere, completely distracted from the lesson.",
        "A face turned to the side with eyes looking at something other than the class content.",
        "A student not paying attention, looking down at phone or away from the computer."
    ],
    [ # bored (4)
        "A student with a blank, expressionless face showing complete boredom during class.",
        "A close-up of a bored face with glazed-over eyes and no interest in the lesson.",
        "A student with chin resting on hand, staring blankly with a listless expression.",
        "A face showing complete disinterest with half-open eyes and a flat expression.",
        "A student with a vacant stare and slack facial muscles, mentally checked out of class."
    ],
    [ # drowsy (5)
        "A student with heavy drooping eyelids, struggling to keep eyes open during class.",
        "A close-up of a drowsy face mid-yawn with watery eyes during an online lesson.",
        "A student with eyes nearly closed, head starting to drop from sleepiness.",
        "A face showing extreme tiredness with slow blinking and a slack jaw.",
        "A student fighting sleep with droopy eyes and nodding head during the lecture."
    ]
]

# ============================================================================
# DAiSEE 4-Class Discrete (Boredom, Engagement, Confusion, Frustration)
# ============================================================================
class_names_daisee4 = [
    "Boredom", "Engagement", "Confusion", "Frustration"
]

class_names_with_context_daisee4 = [
    "A bored student in an online class.",
    "An engaged student in an online class.",
    "A confused student in an online class.",
    "A frustrated student in an online class."
]

class_descriptor_daisee4 = [
    "A student is bored: blank expression, glazed eyes, no interest in the lesson, appearing zoned out.",
    "A student is engaged: focused eyes on screen, attentive expression, actively following the lesson.",
    "A student is confused: furrowed brows, tilted head, puzzled expression, trying to understand.",
    "A student is frustrated: tense face, clenched jaw, visible stress and irritation."
]

prompt_ensemble_daisee4 = [
    [ # Boredom (0)
        "A close-up of a student's face with a blank, expressionless look showing complete boredom.",
        "A student with glazed-over eyes and no interest, staring blankly at the screen.",
        "A face showing disengagement with half-open eyes and a flat lifeless expression.",
        "A student with chin resting on hand, looking listless and mentally checked out.",
        "A close-up of a bored face with droopy features and no emotional response to the lesson.",
        "A student showing signs of tedium with slow blinking and a vacant unfocused stare.",
        "A face with a dull expression, mouth slightly open, showing zero interest in class."
    ],
    [ # Engagement (1)
        "A close-up of a student's face with focused alert eyes looking directly at the screen.",
        "A student with an attentive expression, eyes bright and concentrated on the lesson.",
        "A face showing active participation with widened eyes and a slight nod of understanding.",
        "A student with steady eye contact toward the camera and a calm focused expression.",
        "A close-up of an engaged face with raised eyebrows showing curiosity and interest.",
        "A student with clear open eyes watching the screen attentively with a composed expression.",
        "A face showing sustained attention with gentle focused eyes and an alert gaze."
    ],
    [ # Confusion (2)
        "A close-up of a student's face with furrowed brows and a puzzled questioning expression.",
        "A student showing confusion with squinted eyes and a tilted head during online class.",
        "A face with raised eyebrows and an uncertain look, trying to process difficult content.",
        "A student with a perplexed expression and mouth slightly open in bewilderment.",
        "A close-up showing a confused face with creased forehead and narrowed eyes.",
        "A student scratching their head with a bewildered expression during the lesson.",
        "A face showing cognitive struggle with knitted brows and a questioning gaze."
    ],
    [ # Frustration (3)
        "A close-up of a student's face showing visible frustration with clenched jaw and tension.",
        "A student with a stressed and annoyed expression, struggling with course material.",
        "A face with pressed lips and an irritated frown showing anger at the difficulty.",
        "A student showing emotional distress with narrowed eyes and a tense expression.",
        "A close-up of a frustrated face with flared nostrils and a hard stare.",
        "A student with hands on face showing exasperation and overwhelming stress.",
        "A face showing intense displeasure with tight muscles and a deep frown."
    ]
]

# ============================================================
# DAiSEE 4-Class Discrete: Boredom / Engagement / Confusion / Frustration
# ============================================================

class_names_daisee4 = [
    "Boredom (student in online class).",
    "Engagement (student in online class).",
    "Confusion (student in online class).",
    "Frustration (student in online class).",
]

class_names_with_context_daisee4 = [
    "A student showing boredom during an online class.",
    "A student showing engagement during an online class.",
    "A student showing confusion during an online class.",
    "A student showing frustration during an online class.",
]

class_descriptor_daisee4 = [
    "A student with heavy drooping eyelids, mouth open or yawning, gaze wandering away from the camera, chin resting on hand, showing boredom.",
    "A student with open alert eyes directed straight at the camera, upright posture, mouth closed, showing engagement.",
    "A student with furrowed brows, head tilted to one side, eyes squinting or narrowed, mouth slightly open, showing confusion.",
    "A student with pressed lips, clenched jaw, tense facial muscles, narrowed eyes staring hard at the screen, showing frustration.",
]

