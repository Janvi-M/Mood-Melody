import streamlit as st
import base64
import cv2
from io import BytesIO
import time
from PIL import Image
import numpy as np
import os
import torch
import torchaudio
import soundfile as sf
import random
from colorama import Fore, Style, init
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepface import DeepFace
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# Initialize colorama
init(autoreset=True)

# Spotify API credentials
CLIENT_ID = "id"
CLIENT_SECRET = "secret"

# Emotion-to-Music Prompt Mapping for Music Generation
emotion_to_music = {
    "happy": "A vibrant, upbeat jazz melody with cheerful saxophone and bouncy drums.",
    "sad": "A slow, emotional piano ballad with strings and gentle ambient textures.",
    "angry": "A powerful, aggressive metal riff with fast drums and distorted guitars.",
    "anxious": "A tense, pulsating electronic track with suspenseful build-ups.",
    "peaceful": "A soft, ambient instrumental with warm pads and slow melodies.",
    "energetic": "An uplifting EDM track with pulsing beats and bright synths.",
    "romantic": "A smooth, melodic tune with soft acoustic guitar and gentle piano.",
    "mysterious": "An enigmatic soundtrack with ethereal sounds and curious melodies.",
    "nostalgic": "A retro synthwave track that feels like a memory of the '80s.",
    "dreamy": "A slow, floating composition with reverberating piano and soft strings.",
    "joyful": "A fun, playful tune with marimba, ukulele, and light percussion.",
    "content": "A smooth lo-fi chill beat with jazzy keys and subtle vinyl crackles.",
    "fear": "A dark, suspenseful ambient composition with dissonant textures.",
    "disgust": "An unsettling experimental track with unusual sounds and rhythms.",
    "neutral": "A balanced instrumental with moderate energy and relaxed atmosphere."
}

# Emotion mapping for mood shifting in Spotify recommendations
EMOTION_SHIFT = {
    'sad': 'happy',
    'angry': 'calm',
    'fear': 'relaxed',
    'disgust': 'pleasant',
    'neutral': 'happy',
    'anxious': 'peaceful',
}

# Popular languages mapped to search terms for Spotify
LANGUAGE_KEYWORDS = {
    'english': 'english',
    'spanish': 'spanish latino',
    'hindi': 'hindi bollywood',
    'french': 'french',
    'german': 'german deutsch',
    'italian': 'italian italiano',
    'portuguese': 'portuguese brasil',
    'japanese': 'japanese j-pop',
    'korean': 'korean k-pop',
    'chinese': 'chinese mandarin',
    'arabic': 'arabic',
    'russian': 'russian',
    'tamil': 'tamil',
    'telugu': 'telugu',
    'malayalam': 'malayalam',
    'kannada': 'kannada',
    'bengali': 'bengali',
    'punjabi': 'punjabi'
}

# Simplified Mood to audio features mapping for Spotify
MOOD_AUDIO_FEATURES = {
    'sad': {
        'target_energy': 0.3,
        'target_valence': 0.2,
    },
    'happy': {
        'target_energy': 0.7,
        'target_valence': 0.8,
    },
    'calm': {
        'target_energy': 0.2,
        'target_valence': 0.5,
    },
    'relaxed': {
        'target_energy': 0.3,
        'target_valence': 0.6,
    },
    'pleasant': {
        'target_energy': 0.5,
        'target_valence': 0.7,
    },
    'angry': {
        'target_energy': 0.8,
        'target_valence': 0.3,
    },
    'fear': {
        'target_energy': 0.4,
        'target_valence': 0.2,
    },
    'disgust': {
        'target_energy': 0.5,
        'target_valence': 0.3,
    },
    'neutral': {
        'target_energy': 0.5,
        'target_valence': 0.5,
    }
}

positive_emotions = ["peaceful", "joyful", "content", "dreamy"]
all_emotions = list(emotion_to_music.keys())

# Initialize global variables for models
music_gen_model = None
music_gen_processor = None

def detect_emotion():
    """Capture image from webcam and detect the dominant emotion."""
    st.info("Starting camera... Please look at the camera.")
    
    # Display a placeholder for the camera feed
    camera_placeholder = st.empty()
    camera_placeholder.info("Camera initializing...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera. Using 'neutral' as default.")
        return "neutral"
    
    # Give camera time to warm up
    time.sleep(1.5)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Could not capture frame. Using 'neutral' as default.")
        return "neutral"
    
    # Display the captured image
    camera_placeholder.image(frame, channels="BGR", caption="Captured image")
    
    try:
        with st.spinner("Analyzing emotion..."):
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            st.success(f"Detected Emotion: {dominant_emotion}")
            return dominant_emotion
    except Exception as e:
        st.error(f"Error analyzing emotion: {e}")
        return "neutral"

def initialize_spotify():
    """Initialize and test Spotify API connection."""
    try:
        with st.spinner("Connecting to Spotify API..."):
            auth_manager = SpotifyClientCredentials(
                client_id=CLIENT_ID, 
                client_secret=CLIENT_SECRET
            )
            sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test with a simple query
            results = sp.search(q="artist:Taylor Swift", type='artist', limit=1)
            
            if 'artists' in results and results['artists']['items']:
                return sp
            else:
                st.error("Spotify API connection failed: Unexpected response format")
                return None
    except Exception as e:
        st.error(f"Spotify API connection error: {e}")
        return None

def safe_spotify_request(func, **kwargs):
    """Safely make Spotify API requests with error handling."""
    try:
        return func(**kwargs)
    except spotipy.SpotifyException as e:
        st.warning(f"Spotify API error: {e}")
        return None
    except Exception as e:
        st.warning(f"Error with Spotify request: {e}")
        return None

def search_artist_tracks(sp, artist_name, mood=None, language=None, limit=5):
    """Search for tracks by a specific artist, with better handling for mood and language."""
    try:
        # First search for the artist to get proper spelling and ID
        artist_results = safe_spotify_request(
            sp.search,
            q=artist_name, 
            type='artist', 
            limit=1
        )
        
        if not artist_results or not artist_results['artists']['items']:
            st.warning(f"No artist found with name: {artist_name}")
            return []
            
        # Get artist details
        artist = artist_results['artists']['items'][0]
        artist_id = artist['id']
        correct_artist = artist['name']
        
        tracks = []
        
        # APPROACH 1: Get artist's top tracks first
        top_tracks_result = safe_spotify_request(
            sp.artist_top_tracks,
            artist_id=artist_id
        )
        
        top_tracks = top_tracks_result['tracks'] if top_tracks_result else []
        
        # APPROACH 2: If mood specified, try to get artist's tracks with that mood
        if mood:
            # Strategy 1: Search with both artist and mood
            query = f"artist:{correct_artist} {mood}"
            if language and language.lower() in LANGUAGE_KEYWORDS:
                lang_keyword = LANGUAGE_KEYWORDS[language.lower()]
                query += f" {lang_keyword}"
                
            mood_results = safe_spotify_request(
                sp.search,
                q=query, 
                type='track', 
                limit=limit*2
            )
            
            # Add these tracks to our list
            if mood_results and 'tracks' in mood_results:
                for track in mood_results['tracks']['items']:
                    # Only include tracks by our target artist (since search can return similar artists)
                    if any(a['id'] == artist_id for a in track['artists']):
                        tracks.append({
                            'name': track['name'],
                            'artists': ', '.join([a['name'] for a in track['artists']]),
                            'url': track['external_urls']['spotify'],
                            'preview': track.get('preview_url'),
                            'source': 'mood_search'
                        })
            
            if len(tracks) < limit:
                # Try searching directly without using recommendations
                alternative_search_query = f"{mood} {correct_artist}"
                alt_results = safe_spotify_request(
                    sp.search,
                    q=alternative_search_query, 
                    type='track', 
                    limit=limit*2
                )
                
                if alt_results and 'tracks' in alt_results:
                    for track in alt_results['tracks']['items']:
                        if any(a['id'] == artist_id for a in track['artists']):
                            track_url = track['external_urls']['spotify']
                            if not any(t['url'] == track_url for t in tracks):
                                tracks.append({
                                    'name': track['name'],
                                    'artists': ', '.join([a['name'] for a in track['artists']]),
                                    'url': track_url,
                                    'preview': track.get('preview_url'),
                                    'source': 'alternative_search'
                                })
        
        # APPROACH 3: If still not enough tracks or no mood specified, add top tracks
        if len(tracks) < limit and top_tracks:
            for track in top_tracks:
                # Check if this track is already in our list to avoid duplicates
                track_url = track['external_urls']['spotify']
                if not any(t['url'] == track_url for t in tracks):
                    tracks.append({
                        'name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'url': track_url,
                        'preview': track.get('preview_url'),
                        'source': 'top_tracks'
                    })
        
        # Truncate to requested limit
        return tracks[:limit]
        
    except Exception as e:
        st.error(f"Error searching for artist tracks: {e}")
        return []

def analyze_audio_features(sp, track_id):
    """Analyze audio features to determine if a track matches a mood."""
    try:
        features_result = safe_spotify_request(
            sp.audio_features,
            tracks=[track_id]
        )
        
        if not features_result or not features_result[0]:
            return 'unknown'
            
        features = features_result[0]
        
        # Simple mood classification based on valence and energy
        if features['valence'] < 0.4 and features['energy'] < 0.5:
            return 'sad'
        elif features['valence'] > 0.6 and features['energy'] > 0.6:
            return 'happy'
        elif features['energy'] < 0.4 and features['valence'] > 0.4:
            return 'relaxed'
        elif features['energy'] > 0.7 and features['valence'] < 0.4:
            return 'angry'
        else:
            return 'neutral'
    except:
        return 'unknown'

def search_mood_tracks(sp, mood, language=None, limit=5):
    """Search for tracks based on mood and language with improved strategy."""
    try:
        tracks = []
        
        # Strategy 1: Direct search with mood and language
        query = f"{mood} music"
        if language and language.lower() in LANGUAGE_KEYWORDS:
            lang_keyword = LANGUAGE_KEYWORDS[language.lower()]
            query += f" {lang_keyword}"
            
        results = safe_spotify_request(
            sp.search,
            q=query, 
            type='track', 
            limit=limit*2
        )
        
        # Add these tracks to our results
        if results and 'tracks' in results:
            for track in results['tracks']['items']:
                tracks.append({
                    'name': track['name'],
                    'artists': ', '.join([a['name'] for a in track['artists']]),
                    'url': track['external_urls']['spotify'],
                    'preview': track.get('preview_url'),
                    'source': 'mood_search'
                })
            
        # Strategy 2: If not enough tracks, try searching for playlists with the mood
        if len(tracks) < limit:
            playlist_query = f"{mood}"
            if language and language.lower() in LANGUAGE_KEYWORDS:
                lang_keyword = LANGUAGE_KEYWORDS[language.lower()]
                playlist_query += f" {lang_keyword}"
                
            playlist_results = safe_spotify_request(
                sp.search,
                q=playlist_query, 
                type='playlist', 
                limit=3
            )
            
            if playlist_results and 'playlists' in playlist_results:
                playlist_tracks = []
                # Get tracks from the top 3 playlists
                for playlist in playlist_results['playlists']['items']:
                    playlist_id = playlist['id']
                    playlist_items = safe_spotify_request(
                        sp.playlist_items,
                        playlist_id=playlist_id, 
                        limit=limit
                    )
                    
                    if playlist_items and 'items' in playlist_items:
                        for item in playlist_items['items']:
                            if item.get('track') and 'id' in item['track']:
                                track = item['track']
                                playlist_tracks.append({
                                    'name': track['name'],
                                    'artists': ', '.join([a['name'] for a in track['artists']]),
                                    'url': track['external_urls']['spotify'],
                                    'preview': track.get('preview_url'),
                                    'source': 'playlist'
                                })
                
                # Add unique playlist tracks to our results
                for pt in playlist_tracks:
                    if not any(t['url'] == pt['url'] for t in tracks):
                        tracks.append(pt)
        
        # Strategy 3: Try searching for genre
        if len(tracks) < limit:
            genre_query = f"{mood} genre"
            genre_results = safe_spotify_request(
                sp.search,
                q=genre_query, 
                type='track', 
                limit=limit
            )
            
            if genre_results and 'tracks' in genre_results:
                for track in genre_results['tracks']['items']:
                    track_url = track['external_urls']['spotify']
                    if not any(t['url'] == track_url for t in tracks):
                        tracks.append({
                            'name': track['name'],
                            'artists': ', '.join([a['name'] for a in track['artists']]),
                            'url': track_url,
                            'preview': track.get('preview_url'),
                            'source': 'genre_search'
                        })
            
        # Truncate to requested limit
        return tracks[:limit]
        
    except Exception as e:
        st.error(f"Error searching for mood tracks: {e}")
        return []

def load_music_gen_model():
    """Load the MusicGen model for music generation."""
    global music_gen_model, music_gen_processor
    
    with st.spinner("🎶 Loading MusicGen model... (This may take time initially)"):
        music_gen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        music_gen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

def dynamic_transition_path(start_emotion, total_steps=6):
    """Create a more organic and dynamic emotion flow."""
    flow = [start_emotion]
    current_emotion = start_emotion

    for step in range(total_steps - 2):
        if step < total_steps // 2:
            # Random emotion but weighted towards 'nearby' calm or mysterious or romantic moods
            candidates = ["peaceful", "mysterious", "romantic", "nostalgic", "dreamy", current_emotion]
            next_emotion = random.choice(candidates)
        else:
            # Start moving towards positive endings
            next_emotion = random.choices(positive_emotions, weights=[3, 4, 3, 2], k=1)[0]
        flow.append(next_emotion)
        current_emotion = next_emotion

    # End on a strong positive emotion
    flow.append(random.choice(positive_emotions))
    return flow

def generate_music(prompt, chunk_name="chunk.wav", duration_tokens=256):
    """Generate music based on the prompt."""
    global music_gen_model, music_gen_processor
    
    st.info(f"🎵 Generating music for: {prompt}")
    inputs = music_gen_processor(text=prompt, return_tensors="pt")
    with torch.no_grad():
        audio_values = music_gen_model.generate(**inputs, max_new_tokens=duration_tokens)
    torchaudio.save(chunk_name, audio_values[0].cpu(), 16000)
    return chunk_name

def append_music(existing_file, new_file):
    """Append new audio to the existing audio file."""
    if not os.path.exists(existing_file):
        os.rename(new_file, existing_file)
        return existing_file

    existing_audio, sr = sf.read(existing_file)
    new_audio, sr2 = sf.read(new_file)
    assert sr == sr2, "Sampling rates do not match."

    combined_audio = np.concatenate((existing_audio, new_audio))
    sf.write(existing_file, combined_audio, sr)
    os.remove(new_file)
    return existing_file

def dynamic_emotion_music_transition(start_emotion, progress_bar=None):
    """Handle the entire transition from start to positive end."""
    sequence = dynamic_transition_path(start_emotion)
    st.write(f"✨ Emotion Transition Path: {' -> '.join(sequence)}")

    final_music_file = "dynamic_emotion_music.wav"
    
    # If progress bar wasn't provided, create one
    if progress_bar is None:
        progress_bar = st.progress(0)

    for idx, emotion in enumerate(sequence):
        prompt = emotion_to_music.get(emotion.lower(), "A calm, neutral instrumental")
        chunk_file = generate_music(prompt, chunk_name=f"chunk_{idx}.wav", duration_tokens=256)
        final_music_file = append_music(final_music_file, chunk_file)
        
        # Update progress
        progress = (idx + 1) / len(sequence)
        progress_bar.progress(progress)
        st.write(f"✅ Generated segment {idx + 1}/{len(sequence)} for emotion: {emotion.capitalize()}")

    st.success(f"🎉 Final dynamic music created: {final_music_file}")
    return final_music_file

def create_music_player_page(audio_file):
    """Create a dedicated music player page with visualizations."""
    # Create a container for the music player
    music_container = st.container()
    
    with music_container:
        st.subheader("🎵 Your Generated Music 🎵")
        
        # Display music player with auto-play
        with open(audio_file, "rb") as file:
            audio_bytes = file.read()
            
        # Create a custom HTML audio player with autoplay
        autoplay_audio = f"""
        <audio autoplay controls style="width: 100%;">
          <source src="data:audio/wav;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
        """
        st.markdown(autoplay_audio, unsafe_allow_html=True)
        
        # Add visualization options
        st.subheader("Music Visualization")
        viz_type = st.selectbox(
            "Choose visualization type",
            ["Waveform", "Spectrogram", "None"]
        )
        
        if viz_type != "None":
            # Create visualization placeholder
            viz_placeholder = st.empty()
            
            if viz_type == "Waveform":
                # Create a simple waveform visualization
                audio_data, sr = sf.read(audio_file)
                
                # If stereo, take the mean
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Downsample for visualization
                samples = min(10000, len(audio_data))
                indices = np.linspace(0, len(audio_data) - 1, samples, dtype=int)
                audio_sample = audio_data[indices]
                
                # Plot the waveform
                viz_placeholder.line_chart(audio_sample)
                
            elif viz_type == "Spectrogram":
                # Create a simpler representation that works with Streamlit
                audio_data, sr = sf.read(audio_file)
                
                # If stereo, take the mean
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Generate a simple spectrogram-like visualization
                segment_size = 1000  # Size of each segment
                segments = len(audio_data) // segment_size
                
                # Create a matrix of segment energy levels
                energy_matrix = []
                for i in range(segments):
                    segment = audio_data[i*segment_size:(i+1)*segment_size]
                    energy = np.abs(segment)
                    # Create frequency bands (simplified)
                    bands = 10
                    band_size = segment_size // bands
                    band_energies = []
                    for b in range(bands):
                        band_energy = np.mean(energy[b*band_size:(b+1)*band_size])
                        band_energies.append(band_energy)
                    energy_matrix.append(band_energies)
                
                # Convert to DataFrame for Streamlit charting
                import pandas as pd
                df = pd.DataFrame(energy_matrix)
                viz_placeholder.line_chart(df)

        # Add download button
        st.download_button(
            label="Download Your Music",
            data=audio_bytes,
            file_name="your_generated_music.wav",
            mime="audio/wav"
        )
            
        # Music details
        with st.expander("About This Music"):
            st.write("""
            This music was generated using AI based on your emotional state. 
            The composition transitions through different emotional states to create a dynamic musical journey.
            
            You can share this music or download it for personal use.
            """)

def main():
    """Main application flow."""
    # App setup
    st.set_page_config(page_title="Emotion-Driven Music Tool", layout="wide")
    
    # Check if we're in music player mode
    if 'music_player_active' in st.session_state and st.session_state['music_player_active']:
        create_music_player_page(st.session_state['audio_file'])
        if st.button("← Back to Main App"):
            st.session_state['music_player_active'] = False
            st.rerun()
        return
    
    # Main app UI
    st.title("🎵 Emotion-Driven Music Tool")
    st.write("This app detects your emotion and helps you find or generate music based on it.")

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Emotion Detection")
        if st.button("Detect Emotion"):
            emotion = detect_emotion()
            st.session_state['emotion'] = emotion

    if 'emotion' in st.session_state:
        with col2:
            st.subheader("Choose Your Music Experience")
            option = st.radio(
                "What would you like to do?",
                ["Get Spotify Recommendations", "Generate Custom Music"]
            )

            if option == "Get Spotify Recommendations":
                sp = initialize_spotify()
                if sp:
                    st.subheader("Music Preferences")
                    shift_mood = st.checkbox("Shift my mood")
                    artist_name = st.text_input("Preferred Artist (optional)")
                    language_pref = st.text_input("Preferred Language (optional)")

                    if st.button("Get Recommendations"):
                        with st.spinner("Finding songs..."):
                            emotion = st.session_state['emotion']
                            if shift_mood and emotion in EMOTION_SHIFT:
                                target_mood = EMOTION_SHIFT[emotion]
                            else:
                                target_mood = emotion

                            if artist_name:
                                tracks = search_artist_tracks(
                                    sp, 
                                    artist_name, 
                                    mood=target_mood,
                                    language=language_pref
                                )
                            else:
                                tracks = search_mood_tracks(
                                    sp,
                                    target_mood,
                                    language=language_pref
                                )

                            if tracks:
                                st.subheader("📑 Recommended Songs")
                                for track in tracks:
                                    with st.expander(f"{track['name']} by {track['artists']}"):
                                        st.write(f"🎵 [Listen on Spotify]({track['url']})")
                                        if track['preview']:
                                            st.audio(track['preview'])
                            else:
                                st.warning("No songs found matching your criteria.")
                else:
                    st.error("Could not connect to Spotify API. Please check your credentials.")
            else:  # Generate Custom Music
                st.info("The music generator will create a dynamic composition that transitions through different emotional states.")
                
                # Only show generation button if model is loaded or hasn't been loaded yet
                generation_button = st.button("Generate Music")
                
                if generation_button:
                    # Show progress indicators
                    progress_container = st.container()
                    with progress_container:
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        progress_text.text("Setting up...")
                        
                        # Load model if not already loaded
                        if 'music_gen_model' not in st.session_state:
                            progress_text.text("Loading MusicGen model...")
                            load_music_gen_model()
                            st.session_state['music_gen_model'] = True
                        
                        # Generate music with progress updates
                        progress_text.text("Generating music...")
                        output_file = dynamic_emotion_music_transition(
                            st.session_state['emotion'], 
                            progress_bar=progress_bar
                        )
                        
                        if output_file:
                            progress_text.text("Music generation complete!")
                            # Store the file path and activate music player mode
                            st.session_state['audio_file'] = output_file
                            st.session_state['music_player_active'] = True
                            # Rerun the app to show the music player
                            st.rerun()
                        else:
                            st.error("Music generation failed. Please try again.")

if __name__ == "__main__":
    main()
