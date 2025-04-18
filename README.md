# MoodMelody: Emotion-Driven Music Tool 🎵 🎭

MoodMelody uses AI to analyze your emotional state through your webcam and provides music recommendations or generates custom music to match or improve your mood.

## 🌟 Features

- **Emotion Detection**: Analyzes your facial expressions to detect your current emotion using DeepFace
- **Spotify Integration**: Recommends songs from Spotify based on your detected emotion
  - Search by artist
  - Filter by language preference
  - Option to "shift your mood" to more positive emotions
- **AI Music Generation**: Creates custom music compositions that dynamically transition through emotional states
  - Uses Facebook's MusicGen model to generate unique music pieces
  - Creates a musical journey from your current emotion to a more positive state
- **Music Visualization**: View waveform or spectrogram visualizations of your generated music
- **Download Support**: Save your AI-generated music for later use

## 📋 Requirements

```
streamlit
opencv-python
pillow
numpy
torch
torchaudio
soundfile
colorama
spotipy
deepface
transformers
pandas
```

## 🚀 Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mood-melody.git
   cd mood-melody
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Add your Spotify API credentials:
   - Register your app at the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications)
   - Update the `CLIENT_ID` and `CLIENT_SECRET` variables in `mood-melody.py`

4. Run the application:
   ```
   streamlit run mood-melody.py
   ```

## 💻 Usage

1. Launch the application
2. Click "Detect Emotion" to take a photo using your webcam
3. Choose between "Get Spotify Recommendations" or "Generate Custom Music"
4. For Spotify recommendations:
   - Optionally select "Shift my mood" to get more uplifting music
   - Enter an artist name if you have a preference
   - Choose a language if desired
5. For custom music generation:
   - Click "Generate Music" to create a unique composition
   - The app will show progress as it creates each segment
   - Visualize the audio with waveform or spectrogram options
   - Download your music when complete

## 🧠 How It Works

### Emotion Detection
The app uses DeepFace to analyze facial expressions and determine your dominant emotion from categories like happy, sad, angry, etc.

### Music Recommendation
The app maps emotions to musical qualities and searches Spotify for matching songs. If you choose to "shift your mood," it will recommend music with more positive emotional qualities.

### Music Generation
1. Creates a dynamic path of emotions starting from your detected state
2. Gradually transitions toward more positive emotional states
3. Generates musical segments for each emotion using MusicGen
4. Combines segments into a cohesive musical journey
5. Provides visualization tools to see the audio structure

## 🔮 Future Enhancements

- Improved emotional analysis with multiple frames for better accuracy
- Genre preferences for music recommendations
- Social sharing options
- Custom emotion transition paths
- Voice mood detection option


## 👏 Acknowledgements

- [Facebook AI Research](https://ai.facebook.com/) for MusicGen
- [DeepFace](https://github.com/serengil/deepface) for facial emotion recognition
- [Streamlit](https://streamlit.io/) for the web application framework
- [Spotify API](https://developer.spotify.com/documentation/web-api/) for music recommendations
