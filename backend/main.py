import asyncio
import os
import time
import base64
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None

GEMINI_MODEL = "gemini-2.5-flash"
PAUSE_SECONDS = 3.5

DISPLAY_MAP = {
    "Namaste": "Namaste",
    "Help": "Help",
    "Doctor": "Doctor",
    "Medicine": "Medicine",
    "Water": "Water",
    "Hello": "Hello",
    "MyNameIs": "My Name Is",
    "Krishna": "Krishna",
    "Sorry": "Sorry",
}

def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Gemini] API Key missing. Sentences will use local punctuation fallback.")
        return None
    if genai is None:
        print("[Gemini] google-genai package not found. Run `pip install google-genai`.")
        return None
    return genai.Client(api_key=api_key)

gemini_client = create_gemini_client()

app = FastAPI(title="ISL Translator Web App")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"
VIDEO_DIR = BASE_DIR / "frontend" / "Sign_Library"
MODEL_PATH = Path(__file__).parent / "models" / "action_v4.h5"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
else:
    print(f"[Warning] Sign_Library not found at {VIDEO_DIR}. Voice-to-ISL videos won't load.")

actions = np.array(["Namaste", "Help", "Doctor", "Medicine", "Water", "Hello", "MyNameIs", "Krishna", "Sorry", "Neutral"])
threshold = 0.60

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="tanh", input_shape=(30, 144)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.load_weights(str(MODEL_PATH))

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def _extract_keypoints(results) -> np.ndarray:
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

def _decode_base64_frame(payload: str) -> np.ndarray | None:
    encoded = payload
    if "," in payload:
        encoded = payload.split(",", 1)[1]
    try:
        frame_bytes = base64.b64decode(encoded)
        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    except Exception:
        return None

@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/index.html")
async def read_index_explicit() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/about.html")
async def read_about() -> FileResponse:
    return FileResponse(STATIC_DIR / "about.html")

@app.get("/contact.html")
async def read_contact() -> FileResponse:
    return FileResponse(STATIC_DIR / "contact.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    
    sequence = []
    cooldown = 0
    word_buffer = []
    last_detection_time = time.monotonic()
    last_raw_word = ""
    
    consecutive_predictions = []
    REQUIRED_CONSECUTIVE_FRAMES = 2 

    try:
        while True:
            payload = await websocket.receive_text()
            frame = _decode_base64_frame(payload)
            response_payload = {"type": "empty"} 

            if frame is not None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

                if hands_detected:
                    keypoints = _extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict_on_batch(np.expand_dims(sequence, axis=0))[0]
                        best_idx = int(np.argmax(res))
                        confidence = float(res[best_idx])
                        current_action = actions[best_idx]
                        raw_word = str(current_action)
                        formatted_word = DISPLAY_MAP.get(raw_word, raw_word)

                        await websocket.send_json({
                            "type": "telemetry",
                            "predicted_word": formatted_word,
                            "confidence": round(confidence * 100, 1)
                        })

                        if cooldown > 0:
                            cooldown -= 1
                        else:
                            if confidence >= threshold and current_action != "Neutral":
                                consecutive_predictions.append(raw_word)
                                if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_FRAMES:
                                    if len(set(consecutive_predictions)) == 1: 
                                        cooldown = 20
                                        sequence = []
                                        if raw_word != last_raw_word:
                                            word_buffer.append(formatted_word)
                                            last_detection_time = time.monotonic()
                                            last_raw_word = raw_word
                                            response_payload = {"type": "word", "text": formatted_word}
                                        consecutive_predictions.clear()
                                    else:
                                        consecutive_predictions.pop(0) 
                            else:
                                consecutive_predictions.clear()
                else:
                    sequence = []
                    consecutive_predictions.clear()

            pause_duration = time.monotonic() - last_detection_time
            if word_buffer and pause_duration >= PAUSE_SECONDS:
                signed_words = word_buffer.copy()
                prompt = (
                    "Convert these Indian Sign Language keywords into one short, natural, "
                    "grammatically complete English sentence with proper punctuation and capitalization. "
                    f"Return ONLY the sentence, without quotes or explanations. Keywords: {', '.join(signed_words)}"
                )

                print(f"[Gemini] Pause detected ({pause_duration:.1f}s). Keywords: {signed_words}")
                fluent_sentence = ""
                
                if gemini_client is not None:
                    try:
                        gemini_response = await asyncio.to_thread(
                            gemini_client.models.generate_content,
                            model=GEMINI_MODEL,
                            contents=prompt,
                        )
                        fluent_sentence = (gemini_response.text or "").strip().strip('"')
                        print(f"[Gemini] Response: {fluent_sentence!r}")
                    except Exception as error:
                        print(f"[Gemini] Call error: {error}")

                if not fluent_sentence:
                    fluent_sentence = " ".join(signed_words).capitalize()
                    if not fluent_sentence.endswith((".", "!", "?")):
                        fluent_sentence += "."

                response_payload = {"type": "sentence", "text": fluent_sentence}
                word_buffer.clear()
                last_raw_word = ""

            await websocket.send_json(response_payload)
            if response_payload.get("type") == "sentence":
                print(f"[WebSocket] Dispatched Sentence: {response_payload['text']!r}")

    except WebSocketDisconnect:
        pass