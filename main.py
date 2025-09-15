import cv2
import numpy as np
from keras.models import load_model

# Load the face detection model (pre-trained)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load pre-trained emotion model
model = load_model("emotion_model.h5",compile=False)

# Emotion labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# ====== Spotify integration (paste before "Start webcam") ======
import os
import time
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Config: replace or use env vars ---
CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "3f40e4a63c7f49d585c949c953797b90")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "656ac211238044d2a7c33d75f9fdeff9")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:3000/callback")

# Scopes needed to control playback
SCOPE = "user-read-playback-state user-modify-playback-state user-read-currently-playing"

# Create Spotify client (this opens a browser once for OAuth)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".cache-spotify"   # token cache so you don't re-auth each run
))

# Map emotions to Spotify playlist URIs (replace with playlists you like)
emotion_to_playlist = {
    "Happy":    "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",   # Example
    "Sad":      "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
    "Angry":    "spotify:playlist:37i9dQZF1DWY4xHQp97fN6",
    "Surprise": "spotify:playlist:37i9dQZF1DWU8quswnFt3c",
    "Neutral":  "spotify:playlist:37i9dQZF1DX3rxVfibe1L0",
    "Fear":     "spotify:playlist:37i9dQZF1DX4sWSpwq3LiO",
    "Disgust":  "spotify:playlist:37i9dQZF1DX6xOPeSOGone"
}

def get_playable_device_id():
    """Return an active or first available device id, or None."""
    devices = sp.devices().get("devices", [])
    if not devices:
        return None
    # prefer an active device
    for d in devices:
        if d.get("is_active"):
            return d["id"]
    # else pick first non-restricted device
    for d in devices:
        if not d.get("is_restricted"):
            return d["id"]
    return devices[0]["id"]

def play_music_for_emotion(emotion):
    playlist_uri = emotion_to_playlist.get(emotion)
    if not playlist_uri:
        print("No playlist mapped for emotion:", emotion)
        return

    device_id = get_playable_device_id()
    try:
        sp.start_playback(device_id=device_id, context_uri=playlist_uri)
        print(f"🎶 Starting '{emotion}' playlist on device {device_id}")
    except Exception as e:
        # Common causes: no device available or not premium -> fallback to opening URL
        print("Could not start playback via API:", str(e))
        # Convert spotify:playlist:... to https url
        if playlist_uri.startswith("spotify:playlist:"):
            playlist_id = playlist_uri.split(":")[-1]
            url = f"https://open.spotify.com/playlist/{playlist_id}"
        else:
            url = playlist_uri
        print("Opening playlist in web browser as fallback:", url)
        webbrowser.open(url)

# throttle repeated calls: only change music when emotion changes and cooldown passes
last_emotion = None
last_play_time = 0
COOLDOWN_SECONDS = 15   # change as you like (prevents spamming API)
# ================================================================

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Convert to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        # 1. Crop the face
        face_roi = gray[y:y+h, x:x+w]

        # 2. Resize to 48x48 (standard for emotion recognition datasets)
        face_resized = cv2.resize(face_roi, (64, 64))

        # 3. Normalize pixel values (0-1 range)
        face_normalized = face_resized / 255.0

        # 4. (Optional) Save cropped face for dataset creation
        # cv2.imwrite("dataset/face_" + str(count) + ".jpg", face_resized)
        
        # Reshape for model input
        face_input = np.expand_dims(face_normalized, axis=0)   # (1,48,48)
        face_input = np.expand_dims(face_input, axis=-1)       # (1,48,48,1)

        # Predict emotion
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]

                # --- existing: you computed emotion string here ---
        # emotion = emotion_labels[emotion_idx]

        # Trigger Spotify playback (only when emotion changed and cooldown passed)
        now = time.time()
        if (emotion != last_emotion) and (now - last_play_time > COOLDOWN_SECONDS):
            try:
                play_music_for_emotion(emotion)
            except Exception as e:
                print("Playback error:", e)
            last_emotion = emotion
            last_play_time = now

        # Draw rectangle + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0,255,0), 2)

        # Show cropped face (for checking)
        #cv2.imshow("Cropped Face", face_resized)

    # Show the video feed with boxes
    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
 
