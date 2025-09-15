# 🎶 Emotion-to-Music AI
AI project that detects emotions via webcam and recommends music using Spotify API.


## 🚀 Features
- Real-time face emotion detection (CNN model)
- Maps emotions → Spotify playlists
- Works with both **Free** and **Premium** Spotify accounts:
  - Premium → Auto playback
  - Free → Opens playlist in browser (click Play manually)

**📦 Tech Stack**

Programming: Python 3.10+
ML/DL: TensorFlow, Keras, NumPy
Computer Vision: OpenCV (Haar Cascades)
Music Integration: Spotipy, Spotify Web API
Auth: OAuth 2.0 (SpotifyOAuth)
Model: Pre-trained CNN (.h5) on FER-2013

**🔄 Project Flow**

Detect face using OpenCV.
Classify emotion with CNN.
Map emotion → Spotify playlist.
Auto play (Premium) / Open playlist (Free).

## 🛠️ Installation
```bash
git clone https://github.com/safaltasaxena/EmotionToMusic.git
cd emotion-to-music
pip install -r requirements.txt

