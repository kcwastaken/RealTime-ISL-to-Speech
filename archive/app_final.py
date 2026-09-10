import base64
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model

# These packages are only required by the desktop CustomTkinter application.
# Keeping them optional lets Uvicorn serve the web frontend without them.
DESKTOP_IMPORT_ERROR = None
try:
    import customtkinter as ctk
    import pyttsx3
    from PIL import Image
except ModuleNotFoundError as error:
    DESKTOP_IMPORT_ERROR = error
    ctk = None

# Web application served by: uvicorn app_final:app --reload
app = FastAPI()

PROJECT_DIR = Path(__file__).parent
ACTIONS = np.array(["Namaste", "ThankYou", "Sorry", "Hello", "Busy", "Clean"])
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.50
MODEL_PATH = PROJECT_DIR / "action.h5"

# The model expects 30 frames of both hands' x/y/z landmarks: (30, 126).
recognition_model = load_model(MODEL_PATH, compile=False)
if recognition_model.input_shape != (None, SEQUENCE_LENGTH, 126):
    raise RuntimeError(f"Unexpected model input shape: {recognition_model.input_shape}")
if recognition_model.output_shape[-1] != len(ACTIONS):
    raise RuntimeError(
        f"The model has {recognition_model.output_shape[-1]} outputs, "
        f"but {len(ACTIONS)} labels are configured."
    )


def extract_hand_keypoints(results) -> np.ndarray:
    """Return landmarks in precisely the layout used by action.h5 training."""
    left_hand = (
        np.array([[point.x, point.y, point.z] for point in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[point.x, point.y, point.z] for point in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([left_hand, right_hand])


def decode_browser_frame(payload: str) -> np.ndarray | None:
    """Decode the JPEG data URL sent from static/index.html."""
    try:
        encoded_frame = payload.split(",", 1)[1] if "," in payload else payload
        frame_bytes = base64.b64decode(encoded_frame)
        return cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, cv2.error):
        return None


@app.websocket("/ws")
async def translate_websocket(websocket: WebSocket) -> None:
    """Receive browser frames and return stable ISL predictions."""
    await websocket.accept()
    sequence: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    recent_predictions: deque[int] = deque(maxlen=8)
    last_sent_action = ""
    last_sent_at = 0.0
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            frame = decode_browser_frame(await websocket.receive_text())
            response = {"type": "empty"}

            if frame is None:
                await websocket.send_json(response)
                continue

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

            if not hands_detected:
                sequence.clear()
                recent_predictions.clear()
                last_sent_action = ""
            else:
                sequence.append(extract_hand_keypoints(results))

                if len(sequence) == SEQUENCE_LENGTH:
                    scores = recognition_model.predict_on_batch(
                        np.expand_dims(np.asarray(sequence), axis=0)
                    )[0]
                    prediction_index = int(np.argmax(scores))
                    confidence = float(scores[prediction_index])

                    if confidence >= PREDICTION_THRESHOLD:
                        recent_predictions.append(prediction_index)
                        vote_count = recent_predictions.count(prediction_index)
                        current_action = str(ACTIONS[prediction_index])
                        now = time.monotonic()

                        # A majority vote removes flickering predictions.
                        if vote_count >= 5 and (
                            current_action != last_sent_action
                            or now - last_sent_at >= 2.0
                        ):
                            response = {
                                "type": "word",
                                "text": current_action,
                                "confidence": round(confidence * 100),
                            }
                            last_sent_action = current_action
                            last_sent_at = now

            # index.html sends its next frame after receiving this acknowledgement.
            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass
    finally:
        holistic.close()

# Set modern theme
if ctk is not None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

class SignLanguageApp(ctk.CTk if ctk is not None else object):
   def __init__(self):
        super().__init__()

        self.title("ISL Two-Way Translator")
        self.geometry("1000x600")

        # Create a 2-column grid (Left for Video, Right for Chat)
        self.grid_columnconfigure(0, weight=6) # 60% of screen
        self.grid_columnconfigure(1, weight=4) # 40% of screen
        self.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDE: Video Frame ---
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Webcam Feed Loading...")
        self.video_label.pack(expand=True)

        # --- RIGHT SIDE: Chat Frame ---
        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.chat_log = ctk.CTkTextbox(self.chat_frame, font=("Arial", 16))
        self.chat_log.pack(expand=True, fill="both", padx=10, pady=10)
        self.chat_log.insert("0.0", "--- Chat Started ---\n")

        # AI Backend Setup (Optimized for Continuous Sentences)
        self.actions = np.array(["Namaste", "ThankYou", "Sorry", "Hello", "Busy", "Clean", "Neutral"])
        self.threshold = 0.8  # Higher threshold prevents "stuttering" words
        self.sequence = []    # Stores the sliding window of 30 frames
        self.sentence = []    # Stores the words that make up your sentence
        self.last_spoken_action = ""
        # --- NEW: The AI Cooldown Timer ---
        self.cooldown = 0

        # Load AI Models
        self.model = load_model("action.h5")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # TTS setup (runs in daemon thread so UI never blocks)
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.tts_thread.start()

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        # Handle window closing properly
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the video loop
        self.update_video()

   def _voice_worker(self):
        while True:
            text = self.tts_queue.get()
            
            # If we receive "None", it means the app is closing. Shut down the thread.
            if text is None:
                self.tts_queue.task_done()
                break
                
            try:
                # 1. Boot up a completely fresh engine for EVERY word
                engine = pyttsx3.init()
                engine.setProperty("rate", 250)
                
                # 2. Set the voice (Optional, keeps it sounding normal)
                voices = engine.getProperty("voices")
                if len(voices) > 1:
                    engine.setProperty("voice", voices[1].id)

                # 3. Speak the word
                engine.say(text)
                engine.runAndWait()
                
                # 4. CRITICAL: Destroy the engine so Windows doesn't lock it up!
                del engine 

            except Exception as e:
                print(f"Voice engine skipped a word: {e}")
                
            finally:
                self.tts_queue.task_done()

   def _extract_keypoints(self, results):
        # Correctly extracts keypoints from the Holistic model
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([lh, rh])

   def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # --- THE SPEED FIX: Shrink the frame so the AI does 4x less math ---
            frame = cv2.resize(frame, (640, 480))
            # 1. AI MATH (Runs on the original, un-flipped frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            image.flags.writeable = True
            
            # 2. UI VISUALS (Flip the frame horizontally for the mirror effect)
            frame = cv2.flip(frame, 1)
            
            # --- THE STRICT GATEKEEPER & SPAM BLOCKER ---
            hands_detected = results.left_hand_landmarks or results.right_hand_landmarks

            if hands_detected:
                # Extract data from the original AI math
                keypoints = self._extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                # --- NEW: Check if the AI is on Cooldown ---
                if self.cooldown > 0:
                    self.cooldown -= 1  # Count down the timer
                    
                # Once we have 30 frames, make a prediction
                if len(self.sequence) == 30:
                    res = self.model.predict_on_batch(np.expand_dims(self.sequence, axis=0))[0]
                    current_action = self.actions[np.argmax(res)]
                    confidence = res[np.argmax(res)]

                    # Is it confident?
                    if confidence > self.threshold: 
                        
                        # Rule 1: Ignore Neutral transitions
                        if current_action != "Neutral":
                            
                            # Rule 2: Prevent stuttering
                            if len(self.sentence) == 0 or current_action != self.sentence[-1]:
                                
                                self.sentence.append(current_action)
                                
                                # Keep banner to the last 5 words
                                if len(self.sentence) > 5:
                                    self.sentence = self.sentence[-5:]
                                
                                # Print to Chat Log
                                self.chat_log.insert("end", f"User 1: {' '.join(self.sentence)}\n")
                                self.chat_log.see("end") 
                                
                                # Speak the word
                                self.tts_queue.put(current_action)

                                # ---> THE HYBRID FIX: BRING BACK THE MEMORY WIPE <---
                                # Instantly clears the buffer so the AI doesn't read the 
                                # messy "transition" frames when you move to the next sign!
                                self.sequence = []

                        # NOTICE: We NO LONGER wipe `self.sequence = []`!
                # Draw the recognized text ON TOP OF the flipped frame
                cv2.putText(frame, f"Recognized: {self.last_spoken_action}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            else:
                # --- HANDS ARE DOWN ---
                self.sequence = [] 
                self.last_spoken_action = "" 
                cv2.putText(frame, "Recognized: Waiting...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # --- RENDER TO UI ---
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
            self.video_label.configure(image=imgtk, text="")
        
        # Loop again in 15ms
        self.after(10, self.update_video)

   def on_closing(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, "holistic"):
            self.holistic.close()
        if hasattr(self, "tts_queue"):
            self.tts_queue.put(None)
        self.destroy()

STATIC_DIR = Path(__file__).parent / "static"

# Mount the static directory under the /static path so it does not interfere with WebSockets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Explicitly serve the index.html file only for standard HTTP GET requests to the root
@app.get("/")
async def serve_home():
    return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    if DESKTOP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The desktop app requires its optional dependencies to be installed."
        ) from DESKTOP_IMPORT_ERROR
    desktop_app = SignLanguageApp()
    desktop_app.mainloop()
