import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 1. SETUP
actions = np.array(['Namaste', 'Water', 'Goodbye'])
model = load_model('action.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 2. VARIABLES FOR SLIDING WINDOW
sequence = []
sentence = []
threshold = 0.8 # Only show if AI is > 80% confident

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Extract landmarks for current frame
    keypoints = []
    if results.multi_hand_landmarks:
        for res in results.multi_hand_landmarks[0].landmark:
            keypoints.append([res.x, res.y, res.z])
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
    else:
        keypoints = [[0,0,0]] * 21

    # Add current frame to our "30-frame window"
    sequence.append(np.array(keypoints).flatten())
    sequence = sequence[-30:] # Keep only the last 30 frames
    
    # 3. PREDICTION LOGIC
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
        
        # 4. VIZUALIZATION
        if confidence > threshold:
            if len(sentence) > 0:
                if action != sentence[-1]: # Avoid repeating the same word
                    sentence.append(action)
            else:
                sentence.append(action)

        if len(sentence) > 5: sentence = sentence[-5:] # Keep last 5 words

        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('ISL Real-time LSTM', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()