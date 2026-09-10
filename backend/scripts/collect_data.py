import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

# Paths & Setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "mp_data"

actions = np.array(["Namaste", "Help", "Doctor", "Medicine", "Water", "Hello", "MyNameIs", "Krishna", "Sorry", "Neutral"])
no_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(DATA_PATH / action / str(sequence), exist_ok=True)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results) -> np.ndarray:
    # 1. Left Hand: Wrist shift & bounding box scaling
    lh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh_coords = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                            for res in results.left_hand_landmarks.landmark])
        max_val = np.max(np.abs(lh_coords))
        if max_val > 0:
            lh_coords = lh_coords / max_val
        lh = lh_coords.flatten()

    # 2. Right Hand: Wrist shift & bounding box scaling
    rh = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh_coords = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                            for res in results.right_hand_landmarks.landmark])
        max_val = np.max(np.abs(rh_coords))
        if max_val > 0:
            rh_coords = rh_coords / max_val
        rh = rh_coords.flatten()

    # 3. Pose: Vectors scaled by shoulder width
    pose_features = np.zeros(18)
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        nose = np.array([pose[0].x, pose[0].y, pose[0].z])
        l_sh = np.array([pose[11].x, pose[11].y, pose[11].z])
        r_sh = np.array([pose[12].x, pose[12].y, pose[12].z])
        l_wr = np.array([pose[15].x, pose[15].y, pose[15].z])
        r_wr = np.array([pose[16].x, pose[16].y, pose[16].z])

        lw_nose = l_wr - nose
        lw_lsh = l_wr - l_sh
        lw_rsh = l_wr - r_sh
        rw_nose = r_wr - nose
        rw_lsh = r_wr - l_sh
        rw_rsh = r_wr - r_sh
        raw_pose = np.concatenate([lw_nose, lw_lsh, lw_rsh, rw_nose, rw_lsh, rw_rsh])

        shoulder_width = np.linalg.norm(l_sh - r_sh)
        if shoulder_width > 0:
            pose_features = raw_pose / shoulder_width
        else:
            pose_features = raw_pose

    return np.concatenate([lh, rh, pose_features])

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw Landmarks
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Visual cues for recording intervals
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting: {action} | Video #{sequence + 1}', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting: {action} | Video #{sequence + 1}', (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                keypoints = extract_keypoints(results)
                npy_path = DATA_PATH / action / str(sequence) / f"{frame_num}.npy"
                np.save(str(npy_path), keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()