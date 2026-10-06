# Real-Time Indian Sign Language Translator

A real-time **Indian Sign Language (ISL) to Speech Translator** that uses computer vision and deep learning to recognize signs through a webcam and convert them into text and speech.

## Features

* Real-time ISL recognition using a webcam
* Hand, pose, and facial landmark detection
* LSTM-based sign sequence classification
* Conversion of recognized signs into text
* Text-to-speech support
* Interactive web-based interface

## Tech Stack

* **Frontend:** HTML, JavaScript, Tailwind CSS
* **Backend:** Python, FastAPI, WebSockets
* **Computer Vision:** MediaPipe Holistic
* **Deep Learning:** TensorFlow, Keras, LSTM
* **AI:** Gemini API
* **Speech:** Web Speech API

## How It Works

1. The webcam captures the user's signing in real time.
2. MediaPipe Holistic extracts hand, pose, and facial landmarks.
3. Landmark sequences are processed by the trained LSTM model.
4. The model predicts the corresponding ISL sign.
5. The recognized sign is converted into text and speech.

## Installation

Clone the repository:

```bash
git clone https://github.com/kcwastaken/RealTime-ISL-to-Speech.git
cd RealTime-ISL-to-Speech
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open the frontend in a browser and allow camera access to start the real-time translator.

## Dataset

The model was trained using **self-collected Indian Sign Language video data**. No external image dataset was used.

## Purpose

This project aims to make communication more accessible by providing a real-time interface for translating Indian Sign Language into spoken communication.
