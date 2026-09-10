import cv2
import numpy as np
import os
import mediapipe as mp
import time

DATA_PATH = os.path.join('data', 'mp_data')
actions = np.array(["Namaste", "Help", "Doctor", "Medicine", "Water", "Hello", "MyNameIs", "Krishna", "Sorry", "Neutral"])
no_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        wrist = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.left_hand_landmarks.landmark]).flatten()
        
    rh = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        wrist = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - wrist.x, res.y - wrist.y, res.z - wrist.z] 
                       for res in results.right_hand_landmarks.landmark]).flatten()

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
        
        pose_features = np.concatenate([lw_nose, lw_lsh, lw_rsh, rw_nose, rw_lsh, rw_rsh])

    return np.concatenate([lh, rh, pose_features])

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                if frame_num == 0:
                    cv2.putText(frame, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(frame, f'Collecting frames for {action} Video Number {sequence}', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
cap.release()
cv2.destroyAllWindows()