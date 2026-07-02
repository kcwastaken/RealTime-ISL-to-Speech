import speech_recognition as sr
import cv2
import os
import pyttsx3 # <--- NEW: Voice library

# <--- NEW: Initialize the voice box once at the top
engine = pyttsx3.init()
engine.setProperty('rate', 150) 

def play_video(action):
    # (STAYS THE SAME - Your existing video player code)
    video_path = f'Sign_Library/{action}.mp4'
    if not os.path.exists(video_path): return
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Sign Output', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyWindow('Sign Output')

def listen_for_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio).lower()
        print(f"I heard: {text}")

        # <--- NEW: Computer speaks back to confirm
        engine.say(f"Showing sign for {text}")
        engine.runAndWait()

        # (STAYS THE SAME - Your existing word matching)
        if "namaste" in text:
            play_video("Namaste")
        elif "thank you" in text:
            play_video("ThankYou")
        elif "sorry" in text:
            play_video("Sorry")
         elif "busy" in text:
            play_video("busy")
         elif "hello" in text:
            play_video("hello")
         elif "clean" in text:
            play_video("clean,")

    except Exception as e:
        print("Error understanding audio")

if __name__ == "__main__":
    while True:
        listen_for_voice()