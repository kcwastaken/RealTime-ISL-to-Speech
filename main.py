import time
import requests
import base64
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Configure your API Key here
GEMINI_API_KEY = "AIzaSyB6GtUBYTdgcpwtIpuEO6ClBGZRzmQFZyk"

app = FastAPI(title="ISL Translator Web App")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Model Setup
# 1. Add the new words here:
actions = np.array(["Namaste", "ThankYou", "Sorry", "Hello", "Busy", "Clean", "MyNameIs", "Krishna"])
threshold = 0.8

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))

# 2. Change Dense(6) to Dense(8) because we have 8 words now!
model.add(Dense(8, activation="softmax"))

# 3. Load the new version 2 file!
model.load_weights("action_v2.h5")
# MediaPipe Setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def _extract_keypoints(results) -> np.ndarray:
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([lh, rh])

def _decode_base64_frame(payload: str) -> np.ndarray | None:
    encoded = payload
    if "," in payload:
        encoded = payload.split(",", 1)[1]
    try:
        frame_bytes = base64.b64decode(encoded)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    
    sequence = []
    cooldown = 0
    
    # NLP Sentence Building Variables
    word_buffer = []
    last_detection_time = time.time()
    last_word = ""

    try:
        while True:
            # 1. Receive frame from browser
            payload = await websocket.receive_text()
            frame = _decode_base64_frame(payload)
            
            # Default response to keep the ACK loop moving
            response_payload = {"type": "empty"} 

            if frame is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

                if hands_detected:
                    keypoints = _extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if cooldown > 0:
                        cooldown -= 1

                    # 2. Predict Sign if we have 30 frames
                    if cooldown == 0 and len(sequence) == 30:
                        res = model.predict_on_batch(np.expand_dims(sequence, axis=0))[0]
                        current_action = actions[np.argmax(res)]
                        confidence = float(res[np.argmax(res)])

                        if confidence > threshold and current_action != "Neutral":
                            cooldown = 20
                            sequence = []
                            current_word = str(current_action)

                            # 3. Add to Sentence Buffer if it's a new word
                            if current_word != last_word:
                                word_buffer.append(current_word)
                                last_detection_time = time.time()
                                last_word = current_word
                                response_payload = {"type": "word", "text": current_word}

                else:
                    sequence = [] # Clear sequence if hands disappear

            # 4. THE NLP TRIGGER via Direct HTTP (No Protobuf conflicts!)
            if len(word_buffer) > 0 and (time.time() - last_detection_time) > 4.0:
                prompt = f"The user just signed a sequence of words in Indian Sign Language. Combine these exact keywords into ONE single, cohesive, and fluent English sentence. Do not make a list. Keywords: {', '.join(word_buffer)}"
                
                if GEMINI_API_KEY != "YOUR_API_KEY_HERE":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                    payload_data = {"contents": [{"parts": [{"text": prompt}]}]}
                    
                    try:
                        resp = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload_data)
                        if resp.status_code == 200:
                            result_json = resp.json()
                            fluent_sentence = result_json['candidates'][0]['content']['parts'][0]['text'].strip()
                            response_payload = {"type": "sentence", "text": fluent_sentence}
                        else:
                            print(f"API Error: {resp.text}")
                    except Exception as e:
                        print(f"Request Error: {e}")
                else:
                    print("Please set your GEMINI_API_KEY")
                
                # Reset buffer after generating sentence
                word_buffer = []
                last_word = ""

            # 5. Send exactly ONE response payload to trigger the next frame
            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        pass