import asyncio
import os
import time
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
PAUSE_SECONDS = 1.5

# practice.html's confidence meter and auto-advance-when-matched feature
# both depend on receiving this "telemetry" message every frame - it is a
# real feature, not a debug aid, so it stays on by default. translate.html
# doesn't use it and safely ignores it. Only turn this off if profiling
# shows the extra send_json call is actually a bottleneck, which is
# unlikely compared to the MediaPipe/model calls.
DEBUG_TELEMETRY = True

# Frames with no detected hand are tolerated up to this many times in a row
# before the in-progress 30-frame sequence buffer is thrown away.
MISS_TOLERANCE = 4

# Frames are downscaled to this width before MediaPipe processing.
PROCESS_WIDTH = 480

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
        print("[Gemini] API Key missing. Sentences will use local fallback.")
        return None
    if genai is None:
        return None
    return genai.Client(api_key=api_key)


gemini_client = create_gemini_client()

app = FastAPI(title="ISL Translator Web App")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"
VIDEO_DIR = BASE_DIR / "frontend" / "Sign_Library"
MODEL_PATH = Path(__file__).parent / "models" / "action_v4.h5"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Pages reference their assets as relative paths ("assets/styles.css",
# "assets/site.js", "./assets/avatar.js" in translate.html's module import)
# which resolve to /assets/... at the site root - not /static/assets/...
# This mount is what makes those requests actually resolve instead of 404ing.
ASSETS_DIR = STATIC_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

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
model.predict_on_batch(np.zeros((1, 30, 144), dtype=np.float32))  # warm-up at startup

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _extract_keypoints(results) -> np.ndarray:
    if results.left_hand_landmarks:
        raw = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
        shifted = raw - raw[0]
        max_val = np.max(np.abs(shifted))
        lh = (shifted / max_val).flatten() if max_val > 0 else shifted.flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        raw = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
        shifted = raw - raw[0]
        max_val = np.max(np.abs(shifted))
        rh = (shifted / max_val).flatten() if max_val > 0 else shifted.flatten()
    else:
        rh = np.zeros(21 * 3)

    pose_features = np.zeros(18)
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        nose = np.array([pose[0].x, pose[0].y, pose[0].z])
        l_sh = np.array([pose[11].x, pose[11].y, pose[11].z])
        r_sh = np.array([pose[12].x, pose[12].y, pose[12].z])
        l_wr = np.array([pose[15].x, pose[15].y, pose[15].z])
        r_wr = np.array([pose[16].x, pose[16].y, pose[16].z])

        raw_pose = np.concatenate([l_wr - nose, l_wr - l_sh, l_wr - r_sh, r_wr - nose, r_wr - l_sh, r_wr - r_sh])
        shoulder_width = np.linalg.norm(l_sh - r_sh)
        pose_features = (raw_pose / shoulder_width) if shoulder_width > 0 else raw_pose

    return np.concatenate([lh, rh, pose_features])


def _decode_frame_bytes(payload: bytes) -> np.ndarray | None:
    """Decodes a raw JPEG byte payload sent as a binary WebSocket message.
    No base64/JSON wrapping - this is the ~33% smaller, no-parse-overhead
    path compared to the old base64-in-a-JSON-string transport."""
    try:
        frame_np = np.frombuffer(payload, dtype=np.uint8)
        return cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= PROCESS_WIDTH:
        return frame
    scale = PROCESS_WIDTH / w
    return cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def _process_frame_sync(frame: np.ndarray):
    """Runs on a worker thread via asyncio.to_thread."""
    frame = _downscale(frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return holistic.process(image_rgb)


def _predict_sync(sequence: list) -> np.ndarray:
    return model.predict_on_batch(np.expand_dims(sequence, axis=0))[0]


async def fetch_gemini_sentence(ws: WebSocket, signed_words: list):
    prompt = (
        "Convert these Indian Sign Language keywords into one short, natural, "
        "grammatically complete English sentence with proper punctuation and capitalization. "
        f"Return ONLY the sentence, without quotes or explanations. Keywords: {', '.join(signed_words)}"
    )
    fluent_sentence = ""
    if gemini_client is not None:
        try:
            gemini_response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=prompt,
            )
            fluent_sentence = (gemini_response.text or "").strip().strip('"')
        except Exception as error:
            print(f"[Gemini] Call error: {error}")

    if not fluent_sentence:
        fluent_sentence = " ".join(signed_words).capitalize()
        if not fluent_sentence.endswith((".", "!", "?")):
            fluent_sentence += "."

    try:
        await ws.send_json({"type": "sentence", "text": fluent_sentence})
    except Exception:
        pass


@app.get("/")
async def read_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{filename}.html")
async def read_html(filename: str) -> FileResponse:
    return FileResponse(STATIC_DIR / f"{filename}.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    # --- Latest-frame-only pipeline -------------------------------------
    conn_state = {"payload": None, "disconnected": False}
    new_frame = asyncio.Event()

    async def reader():
        try:
            while True:
                # receive() gives us the raw message so we can branch on
                # binary vs text - lets old base64/text clients keep working
                # during rollout, while new clients use the faster binary path.
                message = await websocket.receive()
                if "bytes" in message and message["bytes"] is not None:
                    conn_state["payload"] = message["bytes"]
                    new_frame.set()
                elif "text" in message and message["text"] is not None:
                    # Legacy fallback: old clients sending base64 text frames.
                    import base64
                    text = message["text"]
                    encoded = text.split(",", 1)[1] if "," in text else text
                    try:
                        conn_state["payload"] = base64.b64decode(encoded)
                        new_frame.set()
                    except Exception:
                        pass
        except WebSocketDisconnect:
            conn_state["disconnected"] = True
            new_frame.set()

    reader_task = asyncio.create_task(reader())
    # ----------------------------------------------------------------------

    sequence = []
    cooldown = 0
    word_buffer = []
    last_detection_time = time.monotonic()

    last_raw_word = ""
    global_last_word = ""  # left as-is

    consecutive_predictions = []
    REQUIRED_CONSECUTIVE_FRAMES = 2
    miss_streak = 0

    try:
        while True:
            await new_frame.wait()
            new_frame.clear()

            if conn_state["disconnected"]:
                break

            payload = conn_state["payload"]
            frame = _decode_frame_bytes(payload) if payload else None
            response_payload = {"type": "empty"}

            if frame is not None:
                results = await asyncio.to_thread(_process_frame_sync, frame)
                hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

                if hands_detected:
                    miss_streak = 0
                    keypoints = _extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = await asyncio.to_thread(_predict_sync, sequence)
                        best_idx = int(np.argmax(res))
                        confidence = float(res[best_idx])
                        current_action = actions[best_idx]
                        raw_word = str(current_action)
                        formatted_word = DISPLAY_MAP.get(raw_word, raw_word)

                        if DEBUG_TELEMETRY:
                            await websocket.send_json({
                                "type": "telemetry",
                                "predicted_word": formatted_word,
                                "confidence": round(confidence * 100, 1)
                            })

                        if cooldown > 0:
                            cooldown -= 1
                        else:
                            if confidence >= threshold:
                                if current_action == "Neutral":
                                    last_raw_word = ""
                                    consecutive_predictions.clear()
                                else:
                                    consecutive_predictions.append(raw_word)
                                    if len(consecutive_predictions) >= REQUIRED_CONSECUTIVE_FRAMES:
                                        if len(set(consecutive_predictions)) == 1:
                                            cooldown = 10
                                            sequence = []

                                            if raw_word != last_raw_word and raw_word != global_last_word:
                                                word_buffer.append(formatted_word)
                                                last_detection_time = time.monotonic()
                                                last_raw_word = raw_word
                                                global_last_word = raw_word
                                                response_payload = {"type": "word", "text": formatted_word}

                                            consecutive_predictions.clear()
                                        else:
                                            consecutive_predictions.pop(0)
                            else:
                                consecutive_predictions.clear()
                else:
                    miss_streak += 1
                    if miss_streak >= MISS_TOLERANCE:
                        sequence = []
                        consecutive_predictions.clear()
                        last_raw_word = ""

            pause_duration = time.monotonic() - last_detection_time
            if word_buffer and pause_duration >= PAUSE_SECONDS:
                signed_words = word_buffer.copy()
                word_buffer.clear()
                asyncio.create_task(fetch_gemini_sentence(websocket, signed_words))

            try:
                await websocket.send_json(response_payload)
            except Exception:
                break

    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass