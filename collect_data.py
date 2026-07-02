import cv2
import mediapipe as mp
import numpy as np
import os

# 1. SETUP
DATA_PATH = os.path.join('MP_Data') 
# ONLY record the missing words right now
ACTIONS = np.array([ "Neutral"])   
NO_SEQUENCES = 30 
SEQUENCE_LENGTH = 30 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# THE ADVANCED FUNCTION: Tracks both hands and pads missing hands with zeros
def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Grab the coordinates
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            # Assign to left or right hand
            if handedness.classification[0].label == 'Right':
                rh = coords
            else:
                lh = coords
    return np.concatenate([lh, rh]) # Returns exactly 126 data points

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if frame_num == 0:
                cv2.putText(frame, 'WAITING - Press K to start', (100,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, f'Action: {action} | Video: {sequence}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
                cv2.imshow('Recorder', frame)
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('k'):
                        break
            else:
                cv2.putText(frame, f'RECORDING: {action} Video {sequence}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow('Recorder', frame)

            # Use our new advanced function
            keypoints = extract_keypoints(results)
            
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()