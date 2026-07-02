import os
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import pyttsx3
from tensorflow.keras.models import load_model


# ---------- Shared state ----------
ACTIONS = np.array(['Namaste', 'ThankYou', 'Sorry', 'Hello', 'Busy', 'Clean'])
VOICE_TEXT_MAP = {
    "namaste": "Namaste",
    "thank you": "ThankYou",
    "sorry": "Sorry",
}
PREDICTION_THRESHOLD = 0.5
SEQUENCE_LEN = 30
KEYPOINT_DIM = 126  # 2 hands * 21 landmarks * (x,y,z)

voice_status = "Listening for Voice..."
last_sign_recognized = "-"

video_request_queue = queue.Queue()
tts_queue = queue.Queue()
stop_event = threading.Event()

# MediaPipe draw styles
HAND_LANDMARK_STYLE = mp.solutions.drawing_utils.DrawingSpec(
    color=(255, 255, 255), thickness=2, circle_radius=2
)
HAND_CONNECTION_STYLE = mp.solutions.drawing_utils.DrawingSpec(
    color=(0, 0, 255), thickness=2, circle_radius=1
)


# ---------- TTS setup ----------
def voice_worker() -> None:
    """Single-thread TTS worker to avoid Windows COM lockups."""
    engine = pyttsx3.init()
   # Speed up the voice (Default is usually 200)
    engine.setProperty('rate', 250)
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)

    while True:
        text = tts_queue.get()
        if text is None:
            tts_queue.task_done()
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            tts_queue.task_done()


def speak_async(text: str) -> None:
    tts_queue.put(text)


# ---------- Sign-to-Text helpers ----------
def extract_keypoints(results) -> np.ndarray:
    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            coords = np.array(
                [[res.x, res.y, res.z] for res in hand_landmarks.landmark]
            ).flatten()
            if handedness.classification[0].label == "Right":
                rh = coords
            else:
                lh = coords
    keypoints = np.concatenate([lh, rh])
    if keypoints.shape[0] != KEYPOINT_DIM:
        return np.zeros(KEYPOINT_DIM)
    return keypoints


# ---------- Voice thread ----------
def voice_listener_loop(stop_signal: threading.Event, request_queue: queue.Queue) -> None:
    """Persistent voice listener that pushes sign-video requests."""
    global voice_status

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while not stop_signal.is_set():
            try:
                voice_status = "Listening for Voice..."
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=4)
                heard = recognizer.recognize_google(audio).lower()
                voice_status = f"Heard: {heard}"

                for phrase, action in VOICE_TEXT_MAP.items():
                    if phrase in heard:
                        request_queue.put(action)
                        voice_status = f"Voice Triggered: {action}"
                        break
            except sr.WaitTimeoutError:
                # Normal while waiting for speech.
                continue
            except sr.UnknownValueError:
                voice_status = "Listening for Voice..."
            except sr.RequestError:
                voice_status = "Voice API unavailable"
            except Exception:
                voice_status = "Voice listener error"


def draw_status_bar(frame: np.ndarray, left_text: str, right_text: str) -> np.ndarray:
    """Add bottom status bar with left/right aligned text."""
    h, w = frame.shape[:2]
    bar_h = 40
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (25, 25, 25), -1)

    cv2.putText(
        frame,
        left_text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    text_size = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    right_x = max(10, w - text_size[0] - 10)
    cv2.putText(
        frame,
        right_text,
        (right_x, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def format_mic_status(status: str) -> str:
    """Normalize status text for UI bar."""
    if status == "Listening for Voice...":
        return "[MIC ACTIVE: Listening...]"
    return f"[MIC ACTIVE: {status}]"


def validate_startup_assets() -> bool:
    """Validate required model and sign-video files before startup."""
    missing = []

    model_path = "action.h5"
    if not os.path.exists(model_path):
        missing.append(model_path)

    for action in ACTIONS:
        video_path = os.path.join("Sign_Library", f"{action}.mp4")
        if not os.path.exists(video_path):
            missing.append(video_path)

    if not missing:
        print("[Startup Check] All required assets found.")
        return True

    print("\n[Startup Check] Missing required files:")
    for item in missing:
        print(f"  - {item}")
    print("\nPlease add the missing files, then run again.")
    return False


def main() -> None:
    global last_sign_recognized

    if not validate_startup_assets():
        return

    # AI setup
    print("--> 1. Loading AI Brain (action.h5)...")
    model = load_model("action.h5")
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    print("--> 2. Starting Voice Engine...")
    tts_thread = threading.Thread(target=voice_worker, daemon=True)
    tts_thread.start()
    # Start persistent voice listener in secondary thread
    listener_thread = threading.Thread(
        target=voice_listener_loop,
        args=(stop_event, video_request_queue),
        daemon=True,
    )
    listener_thread.start()

    sequence = []
    predictions_buffer = []
    last_spoken_action = None
    last_spoken_time = 0
    speak_cooldown = 1.5  # Seconds before repeating the same word

    sign_video_cap = None
    sign_video_name = None

    print("--> 3. Waking up WebCam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("--> ERROR: WebCam failed to open! Check if another app is using it.")
    else:
        print("--> 4. WebCam successfully opened. Entering live feed...")
    if not cap.isOpened():
        print("[Startup Check] Camera not available. Please connect/enable camera and retry.")
        stop_event.set()
        return

    with mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=HAND_LANDMARK_STYLE,
                        connection_drawing_spec=HAND_CONNECTION_STYLE,
                    )

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LEN:]

            # Keep the (30, 126) input shape unchanged
            if len(sequence) == SEQUENCE_LEN:
                model_input = np.expand_dims(np.array(sequence), axis=0)
                res = model.predict(model_input, verbose=0)[0]
                print(f"--> AI THINKS: {ACTIONS[np.argmax(res)]} | CONFIDENCE: {np.max(res):.2f}")
                predicted_idx = int(np.argmax(res))
                current_action = ACTIONS[predicted_idx]
                confidence = res[predicted_idx]
                predictions_buffer.append(predicted_idx)
                predictions_buffer = predictions_buffer[-10:]

                unique_preds, counts = np.unique(predictions_buffer, return_counts=True)
                majority_vote_pos = int(np.argmax(counts))
                majority_idx = int(unique_preds[majority_vote_pos])
                majority_count = int(counts[majority_vote_pos])

                if (
                    majority_idx == predicted_idx
                    and majority_count >= 5
                    and confidence > PREDICTION_THRESHOLD
                ):
                    last_sign_recognized = current_action
                    current_time = time.time()
                    # Only speak if it's a new word, OR if enough time has passed for the same word
                    if current_action != last_spoken_action or (
                        current_time - last_spoken_time
                    ) > speak_cooldown:
                        tts_queue.put(current_action)
                        last_spoken_action = current_action
                        last_spoken_time = current_time

                    cv2.rectangle(frame, (0, 0), (640, 50), (245, 117, 16), -1)
                    cv2.putText(
                        frame,
                        f"{current_action} ({confidence * 100:.0f}%)",
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    last_spoken_action = None

            # Handle queued voice-to-sign requests
            while not video_request_queue.empty():
                requested_action = video_request_queue.get()
                video_path = os.path.join("Sign_Library", f"{requested_action}.mp4")
                if os.path.exists(video_path):
                    if sign_video_cap is not None:
                        sign_video_cap.release()
                    sign_video_cap = cv2.VideoCapture(video_path)
                    sign_video_name = requested_action

            # Render sign-video pop-up, if active
            if sign_video_cap is not None:
                ok, vframe = sign_video_cap.read()
                if ok:
                    cv2.imshow("Sign Output", vframe)
                    if sign_video_name:
                        cv2.setWindowTitle("Sign Output", f"Sign Output - {sign_video_name}")
                else:
                    sign_video_cap.release()
                    sign_video_cap = None
                    sign_video_name = None
                    try:
                        cv2.destroyWindow("Sign Output")
                    except cv2.error:
                        pass

            status_right = f"Last Sign Recognized: {last_sign_recognized}"
            status_left = format_mic_status(voice_status)
            frame = draw_status_bar(frame, status_left, status_right)
            cv2.imshow("Bi-directional Sign Language Bridge", frame)

            # Single waitKey call manages both windows
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q") or key == 27:
                break

    # Cleanup
    stop_event.set()
    tts_queue.put(None)
    cap.release()
    if sign_video_cap is not None:
        sign_video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
