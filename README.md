# 🎵 Emotion-Based Music Recommendation & Generation System  

This script enhances user experience using **computer vision, deep learning, and AI-generated music**.  


## How It Works  

### 1️⃣ Emotion Detection  
- Uses **DeepFace** to analyze real-time webcam footage and detect dominant emotions (e.g., Happy, Sad, Angry, etc.).  
- Displays the detected emotion on the webcam feed.  

### 2️⃣ Music Recommendation from YouTube  
- Uses the **YouTube Data API** to search for music videos based on detected emotions.  
- Fetches and prints the top recommended music video link.  

### 3️⃣ AI-Generated Music using MusicGen  
- Uses **Facebook’s MusicGen model** to generate original instrumental tracks based on emotions.  
- Processes text prompts describing the desired music style and generates an audio waveform.  
- Saves the generated audio as **`generated_music.wav`** and plays it automatically.  

### 4️⃣ Real-Time Processing  
- Continuously captures and analyzes webcam frames in a loop.  
- Updates recommendations and generates new music when the detected emotion changes.  
- The loop stops when the user presses **'q'**.  
