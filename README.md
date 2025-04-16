# Emotion-Driven Music Tool

A Python application that detects your emotions in real-time and provides personalized music experiences based on your current emotional state. The tool offers two primary features:

1. **Spotify Music Recommendations** - Find songs that either match or help shift your current mood
2. **AI Music Generation** - Create custom musical compositions that respond to your emotional state

## Features

### Emotion Detection
- Uses DeepFace for real-time facial emotion recognition
- Detects emotions including happy, sad, angry, anxious, neutral, and others
- Webcam integration for instant emotional analysis

### Spotify Integration
- Searches for tracks based on detected emotion
- Allows optional mood shifting (e.g., from sad to happy)
- Supports artist-specific searches
- Language preferences for international music discovery
- Advanced mood-based audio feature filtering

### AI Music Generation
- Utilizes Facebook's MusicGen model for creating original music
- Generates dynamic emotional transitions in musical compositions
- Creates cohesive musical journeys that evolve from your starting emotion to a positive endpoint
- Adaptive prompt system that translates emotions into musical directions

## Requirements

- Python 3.7+
- PyTorch and TorchAudio
- Transformers library
- DeepFace
- OpenCV
- Spotipy (Spotify API client)
- SoundFile
- Colorama

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install torch torchaudio transformers deepface opencv-python spotipy soundfile colorama
   ```
3. Set up Spotify API credentials:
   - Create a Spotify Developer account and create an application
   - Add your Client ID and Client Secret to the code

## Usage

Run the main script:
```
python emotion_music_tool.py
```

The application will:
1. Access your webcam to detect your current emotion
2. Ask whether you want music recommendations or AI-generated music
3. For recommendations, you can specify artist preferences and language choices
4. For generated music, it will create a dynamic composition based on your emotion

## How It Works

### Emotion-to-Music Mapping
The system uses carefully crafted mappings between emotions and musical characteristics:
- Happy → vibrant jazz with cheerful saxophone
- Sad → slow piano ballad with strings
- Angry → powerful metal with distorted guitars
- Anxious → tense electronic with suspenseful build-ups
- And many more...

### Dynamic Transitions
When generating music, the application creates organic transitions between emotional states, typically leading toward positive emotional endpoints through melodic evolution.

### Mood Shifting
The recommendation engine can intentionally shift your emotional state by finding music that counterbalances detected negative emotions (e.g., calming music for angry states).

## Privacy Note

The emotion detection happens locally on your device - no facial images are stored or transmitted.

## Future Improvements

- Support for multi-person emotion detection
- More advanced music generation parameters
- Integration with additional music streaming services
- Mobile application version
