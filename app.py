import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3 # <--- The new voice library
import threading # <--- To speak without freezing the video

# 1. SETUP VOICE ENGINE
engine = pyttsx3.init()
# slow down the speed (default is usually 200)
engine.setProperty('rate', 150) 

# Change voice: 0 for Male, 1 for Female (depending on your Windows settings)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
def speak(text):
    # We use a function so the voice runs in the background
    engine.say(text)
    engine.runAndWait()

# 2. SETUP AI
actions = np.array(['Namaste', 'ThankYou', 'Sorry'])
model = load_model('action.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh, rh = np.zeros(21*3), np.zeros(21*3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            coords = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Right': rh = coords
            else: lh = coords
    return np.concatenate([lh, rh])

# 3. TRACKING VARIABLES
sequence = []
last_spoken_action = None # To prevent repeating the same word
threshold = 0.8 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:] 
    
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
        
        if confidence > threshold:
            # ONLY SPEAK IF IT'S A NEW WORD
            if action != last_spoken_action:
                # Use threading so the camera doesn't freeze while talking!
                threading.Thread(target=speak, args=(action,)).start()
                last_spoken_action = action
            
            cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
            cv2.putText(frame, f"{action} ({confidence*100:.0f}%)", (10,35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            last_spoken_action = None # Reset if no sign is clear

    cv2.imshow('ISL Sign-to-Audio', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()