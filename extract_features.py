import os
import cv2
import json
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. CONFIG ---
WLASL_JSON_PATH = "./WLASL_dataset/WLASL_v0.3.json" # The official WLASL metadata file
VIDEO_DIR = "./WLASL_dataset/videos"
OUTPUT_DIR = "./holistic_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. BUILD THE ID MAP ---
# This maps '66324' -> 'DRINK'
with open(WLASL_JSON_PATH, 'r') as f:
    wlasl_data = json.load(f)

id_to_word = {}
for entry in wlasl_data:
    word = entry['gloss']
    for instance in entry['instances']:
        video_id = instance['video_id']
        id_to_word[video_id] = word.upper()

# --- 3. INITIALIZE MODELS ---
# (Using the Task API logic from before)
base_options = python.BaseOptions
MODELS = {
    'pose': "./mediapipe_landmark/pose_landmarker_heavy.task",
    'hand': "./mediapipe_landmark/hand_landmarker.task",
    'face': "./mediapipe_landmark/face_landmarker.task"
}

p_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base_options(model_asset_path=MODELS['pose'])))
h_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base_options(model_asset_path=MODELS['hand']), num_hands=2))
f_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=base_options(model_asset_path=MODELS['face'])))

# --- 4. EXTRACTION LOGIC ---
all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

for video_file in tqdm(all_videos):
    video_id = video_file.split('.')[0] # '66324'
    
    if video_id not in id_to_word:
        continue # Skip if ID isn't in metadata
    
    gloss = id_to_word[video_id]
    
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_file))
    video_features = []
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Synchronous Detection
        p_res = p_lm.detect(mp_image)
        h_res = h_lm.detect(mp_image)
        f_res = f_lm.detect(mp_image)
        
        # Pose (33), Face (468), Hands (42)
        pose = [[lm.x, lm.y, lm.z] for lm in p_res.pose_landmarks[0]] if p_res.pose_landmarks else [[0,0,0]]*33
        face = [[lm.x, lm.y, lm.z] for lm in f_res.face_landmarks[0]] if f_res.face_landmarks else [[0,0,0]]*468
        
        lh, rh = [[0,0,0]]*21, [[0,0,0]]*21
        if h_res.hand_landmarks:
            for i, landmarks in enumerate(h_res.hand_landmarks):
                side = h_res.handedness[i][0].category_name
                pts = [[lm.x, lm.y, lm.z] for lm in landmarks]
                if side == "Left": lh = pts
                else: rh = pts
        
        video_features.append({"landmarks": np.array(pose + face + lh + rh).flatten().tolist()})
        
    cap.release()

    # SAVE FILENAME AS: GLOSS_ID.json (e.g., DRINK_66324.json)
    output_filename = f"{gloss}_{video_id}.json"
    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        json.dump(video_features, f)