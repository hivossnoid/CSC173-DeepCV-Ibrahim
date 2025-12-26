import torch
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime

# Import the specific MediaPipe Task modules
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORT YOUR MODEL ARCHITECTURE ---
# Ensure train_tgcn.py is in the same folder
from train_tgcn import TGCN, get_adjacency_matrix

# --- 1. CONFIGURATION ---
MODEL_PATH = "FULL_TGCN_holistic.pth"
POSE_TASK = "./mediapipe_landmark/pose_landmarker_heavy.task"
HAND_TASK = "./mediapipe_landmark/hand_landmarker.task"
FACE_TASK = "./mediapipe_landmark/face_landmarker.task"

# --- 2. LOAD BUNDLED MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Map indices to words from your trained model
id_to_word = {i: word for i, word in enumerate(checkpoint['classes'])}
num_classes = len(id_to_word)
num_nodes = 543 # (33 pose + 468 face + 21 LH + 21 RH)
seq_len = checkpoint.get('seq_len', 48)

print(f"✅ Loaded model for {num_classes} classes.")

# Reconstruct Adjacency Matrix and Model
adj = get_adjacency_matrix(num_nodes=num_nodes).to(device)
model = TGCN(num_nodes=num_nodes, num_classes=num_classes, adj=adj)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()

# --- 3. MEDIAPIPE SETUP (SYNC IMAGE MODE) ---
BaseOptions = python.BaseOptions
VisionRunningMode = vision.RunningMode

p_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_TASK), running_mode=VisionRunningMode.IMAGE))
h_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK), running_mode=VisionRunningMode.IMAGE, num_hands=2))
f_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_TASK), running_mode=VisionRunningMode.IMAGE))

# --- 4. INFERENCE LOOP ---
cap = cv2.VideoCapture(0)
frame_window = deque(maxlen=seq_len)
prediction_label = "Waiting..."
confidence = 0.0
last_printed_word = ""
PRINT_THRESHOLD = 0.70  

print("\n--- 🚀 Translation Stream Started (Press 'q' to quit) ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # MediaPipe requires RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Run Inference on frame
    pose_res = p_lm.detect(mp_image)
    hand_res = h_lm.detect(mp_image)
    face_res = f_lm.detect(mp_image)

    # --- 5. STITCH AND SLICE LANDMARKS (FORCE 543 NODES) ---
    # Pose: Ensure 33 nodes
    if pose_res.pose_landmarks:
        pose = [[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0][:33]]
    else:
        pose = [[0.0, 0.0, 0.0]] * 33

    # Face: Ensure 468 nodes (MediaPipe V2 can sometimes return 478)
    
    if face_res.face_landmarks:
        face = [[lm.x, lm.y, lm.z] for lm in face_res.face_landmarks[0][:468]]
    else:
        face = [[0.0, 0.0, 0.0]] * 468
    
    # Hands: Ensure 21 nodes per hand
    
    lh, rh = [[0.0, 0.0, 0.0]] * 21, [[0.0, 0.0, 0.0]] * 21
    if hand_res.hand_landmarks:
        for i, landmarks in enumerate(hand_res.hand_landmarks):
            side = hand_res.handedness[i][0].category_name
            pts = [[lm.x, lm.y, lm.z] for lm in landmarks[:21]]
            if side == "Left": lh = pts
            else: rh = pts
    
    # Concatenate and flatten to (1629,)
    current_frame_flat = np.array(pose + face + lh + rh, dtype=np.float32).flatten()
    frame_window.append(current_frame_flat)

    # --- 6. PREDICTION LOGIC ---
    if len(frame_window) == seq_len:
        # Construct input batch: (1, 48, 1629)
        inp_array = np.stack(list(frame_window))
        inp = torch.from_numpy(inp_array).view(seq_len, num_nodes, 3).to(device)
        
        # Scale-Invariant Normalization (Match Training)
        anchor = inp[:, 0:1, :] # Nose
        shoulder_dist = torch.dist(inp[0, 11, :], inp[0, 12, :]) + 1e-6
        inp_norm = ((inp - anchor) / shoulder_dist).view(1, seq_len, -1)

        with torch.no_grad():
            output = model(inp_norm)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
            prediction_label = id_to_word.get(idx.item(), "Unknown")
            confidence = conf.item()

            # Clean Terminal Printing
            if confidence >= PRINT_THRESHOLD:
                if prediction_label != last_printed_word:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] >> {prediction_label.upper()} ({confidence*100:.1f}%)")
                    last_printed_word = prediction_label

    # --- 7. UI OVERLAY ---
    color = (0, 255, 0) if confidence > PRINT_THRESHOLD else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1) 
    cv2.putText(frame, f"WORD: {prediction_label.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"CONF: {confidence:.2f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Glossa Vision Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()