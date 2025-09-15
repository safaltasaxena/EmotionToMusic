# 🎶 Emotion-to-Music AI
AI project that detects emotions via webcam and recommends music using Spotify API.


## 🚀 Features
- Real-time face emotion detection (CNN model)
- Maps emotions → Spotify playlists
- Works with both **Free** and **Premium** Spotify accounts:
  - Premium → Auto playback
  - Free → Opens playlist in browser (click Play manually)


📦 Tech Stack
🖥️ Programming

Python 3.10+ – Core language used for model + Spotify API integration

🤖 Machine Learning / Deep Learning

TensorFlow / Keras – Emotion recognition model (CNN)

NumPy – Numerical operations & preprocessing

OpenCV (opencv-python) – Real-time face detection using Haar cascades

🎶 Music & API Integration

Spotipy – Python client for Spotify Web API

Spotify Web API – Used to control music playback & fetch playlists

🔒 Authentication

OAuth 2.0 (SpotifyOAuth) – For secure Spotify login & permissions

🗄️ Data / Model

Haar Cascades – Pre-trained XML for detecting faces
Custom Trained Emotion Model (.h5) – CNN trained on FER-2013 dataset (or your dataset)

## 🛠️ Installation
```bash
git clone https://github.com/safaltasaxena/EmotionToMusic.git
cd emotion-to-music
pip install -r requirements.txt

