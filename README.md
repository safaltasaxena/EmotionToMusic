# 🎶 Emotion-to-Music AI
AI project that detects emotions via webcam and recommends music using Spotify API.


## 🚀 Features
- Real-time face emotion detection (CNN model)
- Maps emotions → Spotify playlists
- Works with both **Free** and **Premium** Spotify accounts:
  - Premium → Auto playback
  - Free → Opens playlist in browser (click Play manually)

**📦 Tech Stack**

Programming: Python 3.10+\n
ML/DL: TensorFlow, Keras, NumPy\n
Computer Vision: OpenCV (Haar Cascades)\n
Music Integration: Spotipy, Spotify Web API\n
Auth: OAuth 2.0 (SpotifyOAuth)\n
Model: Pre-trained CNN (.h5) on FER-2013\n

**🔄 Project Flow**

Detect face using OpenCV.\n
Classify emotion with CNN.\n
Map emotion → Spotify playlist.\n
Auto play (Premium) / Open playlist (Free).\n

## 🛠️ Installation
```bash
git clone https://github.com/safaltasaxena/EmotionToMusic.git
cd emotion-to-music
pip install -r requirements.txt

