# 🎶 Emotion-to-Music AI
AI project that detects emotions via webcam and recommends music using Spotify API.


## 🚀 Features
- Real-time face emotion detection (CNN model)
- Maps emotions → Spotify playlists
- Works with both **Free** and **Premium** Spotify accounts:
  - Premium → Auto playback
  - Free → Opens playlist in browser (click Play manually)

**📦 Tech Stack**

Programming: Python 3.10+<br>
ML/DL: TensorFlow, Keras, NumPy<br>
Computer Vision: OpenCV (Haar Cascades)<br>
Music Integration: Spotipy, Spotify Web API<br>
Auth: OAuth 2.0 (SpotifyOAuth)<br>
Model: Pre-trained CNN (.h5) on FER-2013<br>

**🔄 Project Flow**

Detect face using OpenCV.<br>
Classify emotion with CNN.<br>
Map emotion → Spotify playlist.<br>
Auto play (Premium) / Open playlist (Free).<br>

## 🛠️ Installation
```bash
git clone https://github.com/safaltasaxena/EmotionToMusic.git
cd emotion-to-music
pip install -r requirements.txt

