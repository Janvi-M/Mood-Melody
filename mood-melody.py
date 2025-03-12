import cv2
import torch
import torchaudio
import os
import numpy as np
from deepface import DeepFace
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from googleapiclient.discovery import build

# Load the MusicGen model
print("Loading MusicGen model... (This may take time initially)")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
print("Model loaded successfully!")

# YouTube API Setup (Replace with your API Key)
YOUTUBE_API_KEY = "YOUR API KEY"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Emotion-to-Music Mapping
emotion_to_music = {
    "happy": "A vibrant, fast-paced jazz melody with saxophone and energetic percussion",
    "sad": "A slow, emotional orchestral piece with a deep cello and piano",
    "angry": "An intense, heavy metal track with distorted electric guitar and fast drums",
    "neutral": "A peaceful lo-fi beat with soft piano and relaxing synths",
    "surprise": "An unpredictable electronic track with sudden tempo changes and playful melodies"
}

# Emotion-to-YouTube Query Mapping
emotion_to_query = {
    "happy": "happy upbeat music",
    "sad": "sad emotional piano music",
    "angry": "intense rock metal music",
    "neutral": "peaceful lo-fi beat with soft piano",
    "surprise": "mysterious cinematic music"
}

def search_youtube(emotion):
    """Fetches top YouTube music video for the given emotion."""
    query = emotion_to_query.get(emotion.lower(), "calm instrumental music")
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=1,
        type="video"
    )
    response = request.execute()
    
    if response["items"]:
        video_id = response["items"][0]["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        return video_url
    return "No YouTube results found."

def generate_music(emotion):
    """Generates music based on detected emotion."""
    prompt = emotion_to_music.get(emotion.lower(), "A calm, neutral instrumental")
    inputs = processor(text=prompt, return_tensors="pt")
    
    print(f"🎵 Generating music for: {emotion.capitalize()}...")
    
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=256)
    
    output_file = "generated_music.wav"
    torchaudio.save(output_file, audio_values[0].cpu(), 16000)
    print(f"🎶 Music generated and saved as: {output_file}")
    return output_file

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

prev_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotions = analysis[0]['emotion']
        emotion = max(emotions, key=emotions.get)
    except:
        emotion = "Unknown"
    
    cv2.putText(frame, f"Emotion: {emotion.capitalize()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Emotion Recognition ", frame)

    if emotion != "Unknown" and emotion != prev_emotion:
        prev_emotion = emotion
        
        # Recommend YouTube music
        youtube_link = search_youtube(emotion)
        print(f" Recommended YouTube Music: {youtube_link}")
        
        # Generate AI music (optional, can be removed if only using YouTube)
        music_file = generate_music(emotion)
        os.system(f"start {music_file}" if os.name == "nt" else f"afplay {music_file}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
