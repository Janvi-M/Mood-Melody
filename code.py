import torch
import torchaudio
import os
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import time
import random
from colorama import Fore, Style, init
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from deepface import DeepFace
import cv2

init(autoreset=True)

# Spotify API credentials
CLIENT_ID = "client_id"
CLIENT_SECRET = "client_secret"

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

# Mood to audio features mapping for Spotify
MOOD_AUDIO_FEATURES = {
    'sad': {
        'target_energy': 0.3,
        'target_valence': 0.2,
        'max_tempo': 100
    },
    'happy': {
        'target_energy': 0.7,
        'target_valence': 0.8,
        'min_tempo': 100
    },
    'calm': {
        'target_energy': 0.2,
        'target_valence': 0.5,
        'max_tempo': 90
    },
    'relaxed': {
        'target_energy': 0.3,
        'target_valence': 0.6,
        'max_tempo': 95
    },
    'pleasant': {
        'target_energy': 0.5,
        'target_valence': 0.7,
        'target_tempo': 105
    },
    'angry': {
        'target_energy': 0.8,
        'target_valence': 0.3,
        'min_tempo': 120
    },
    'fear': {
        'target_energy': 0.4,
        'target_valence': 0.2,
        'min_instrumentalness': 0.3
    },
    'disgust': {
        'target_energy': 0.5,
        'target_valence': 0.3,
        'target_tempo': 100
    },
    'neutral': {
        'target_energy': 0.5,
        'target_valence': 0.5,
        'target_tempo': 110
    }
}

positive_emotions = ["peaceful", "joyful", "content", "dreamy"]
all_emotions = list(emotion_to_music.keys())
music_gen_model = None
music_gen_processor = None

def detect_emotion():
    """Capture image from webcam and detect the dominant emotion."""
    print(Fore.YELLOW + "Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(Fore.RED + "Error: Could not open camera. Using 'neutral' as default.")
        return "neutral"
    
    # Give camera time to warm up
    time.sleep(1.5)
    
    print(Fore.CYAN + "Capturing emotion. Please look at the camera...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(Fore.RED + "Error: Could not capture frame. Using 'neutral' as default.")
        return "neutral"
    
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        print(Fore.GREEN + f"Detected Emotion: {dominant_emotion}")
        return dominant_emotion
    except Exception as e:
        print(Fore.RED + f"Error analyzing emotion: {e}")
        return "neutral"

def initialize_spotify():
    """Initialize and test Spotify API connection."""
    print(Fore.YELLOW + "Connecting to Spotify API...")
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID, 
            client_secret=CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Test with a simple query
        results = sp.search(q="artist:Taylor Swift", type='artist', limit=1)
        
        if 'artists' in results and results['artists']['items']:
            print(Fore.GREEN + "✅ Spotify API connection successful")
            return sp
        else:
            print(Fore.RED + "❌ Spotify API connection failed: Unexpected response format")
            return None
    except Exception as e:
        print(Fore.RED + f"❌ Spotify API connection error: {e}")
        return None

def search_artist_tracks(sp, artist_name, mood=None, language=None, limit=5):
    """Search for tracks by a specific artist, with better handling for mood and language."""
    try:
        # First search for the artist to get proper spelling and ID
        artist_results = sp.search(q=artist_name, type='artist', limit=1)
        
        if not artist_results['artists']['items']:
            print(Fore.RED + f"No artist found with name: {artist_name}")
            return []
            
        # Get artist details
        artist = artist_results['artists']['items'][0]
        artist_id = artist['id']
        correct_artist = artist['name']
        print(Fore.GREEN + f"Found artist: {correct_artist} (ID: {artist_id})")
        
        tracks = []
        
        # APPROACH 1: Get artist's top tracks first
        top_tracks = sp.artist_top_tracks(artist_id)['tracks']
        
        # APPROACH 2: If mood specified, try to get artist's tracks with that mood
        if mood:
            # Strategy 1: Search with both artist and mood
            query = f"artist:{correct_artist} {mood}"
            if language and language.lower() in LANGUAGE_KEYWORDS:
                lang_keyword = LANGUAGE_KEYWORDS[language.lower()]
                query += f" {lang_keyword}"
                
            print(Fore.CYAN + f"Searching with query: {query}")
            mood_results = sp.search(q=query, type='track', limit=limit*2)
            
            # Add these tracks to our list
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
                print(Fore.CYAN + f"Getting recommendations based on artist and mood...")
                
                # Get seed tracks from the artist's top tracks
                seed_tracks = [track['id'] for track in top_tracks[:2]] if top_tracks else []
                
                if seed_tracks:
                    # Get mood audio features
                    mood_params = MOOD_AUDIO_FEATURES.get(mood.lower(), {})
                try:
                    # Get recommendations using the artist, seed tracks, and mood parameters
                    recs = sp.recommendations(seed_artists=[artist_id], 
                                            seed_tracks=seed_tracks[:2],  # Maximum 5 seeds total (artists + tracks)
                                            limit=limit*2,
                                            **mood_params)
                    
                    # Add recommendations to our list
                    for track in recs['tracks']:
                        # Only include tracks by our target artist
                        if any(a['id'] == artist_id for a in track['artists']):
                            tracks.append({
                                'name': track['name'],
                                'artists': ', '.join([a['name'] for a in track['artists']]),
                                'url': track['external_urls']['spotify'],
                                'preview': track.get('preview_url'),
                                'source': 'recommendations'
                            })

                except Exception as e:
                    print(Fore.RED + f"Error getting recommendations: {e}")
                    print(Fore.YELLOW + "Falling back to top tracks...")
        
        # APPROACH 3: If still not enough tracks or no mood specified, add top tracks
        if len(tracks) < limit:
            print(Fore.CYAN + f"Adding artist's top tracks...")
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
        print(Fore.RED + f"Error searching for artist tracks: {e}")
        return []

def analyze_audio_features(sp, track_id):
    """Analyze audio features to determine if a track matches a mood."""
    try:
        features = sp.audio_features(track_id)[0]
        
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
            
        print(Fore.CYAN + f"Searching for: {query}")
        results = sp.search(q=query, type='track', limit=limit*2)
        
        # Add these tracks to our results
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
            print(Fore.CYAN + f"Searching for playlists with the mood: {mood}")
            playlist_query = f"{mood}"
            if language and language.lower() in LANGUAGE_KEYWORDS:
                lang_keyword = LANGUAGE_KEYWORDS[language.lower()]
                playlist_query += f" {lang_keyword}"
                
            playlist_results = sp.search(q=playlist_query, type='playlist', limit=3)
            
            playlist_tracks = []
            # Get tracks from the top 3 playlists
            for playlist in playlist_results['playlists']['items']:
                playlist_id = playlist['id']
                playlist_items = sp.playlist_items(playlist_id, limit=limit)
                
                for item in playlist_items['items']:
                    if item['track'] and 'id' in item['track']:
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
        
        # Strategy 3: Use recommendations API with mood audio features
        if len(tracks) < limit and mood.lower() in MOOD_AUDIO_FEATURES:
            print(Fore.CYAN + f"Getting recommendations based on mood audio features...")
            
            # Use the first track we found as a seed
            seed_tracks = []
            if tracks:
                track_id = tracks[0]['url'].split('/')[-1]
                seed_tracks = [track_id]
            
            # Get recommendations based on audio features for the mood
            mood_params = MOOD_AUDIO_FEATURES.get(mood.lower(), {})
            recs = sp.recommendations(
                seed_tracks=seed_tracks[:1] if seed_tracks else [],
                seed_genres=['pop'] if not seed_tracks else [],
                limit=limit,
                **mood_params
            )
            
            # Add recommendations to our results
            for track in recs['tracks']:
                track_url = track['external_urls']['spotify']
                if not any(t['url'] == track_url for t in tracks):
                    tracks.append({
                        'name': track['name'],
                        'artists': ', '.join([a['name'] for a in track['artists']]),
                        'url': track_url,
                        'preview': track.get('preview_url'),
                        'source': 'recommendations'
                    })
            
        # Truncate to requested limit
        return tracks[:limit]
        
    except Exception as e:
        print(Fore.RED + f"Error searching for mood tracks: {e}")
        return []

def load_music_gen_model():
    """Load the MusicGen model for music generation."""
    global music_gen_model, music_gen_processor
    
    print(Fore.CYAN + "🎶 Loading MusicGen model... (This may take time initially)")
    music_gen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    music_gen_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    print(Fore.GREEN + "✅ Model loaded successfully!\n")

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
    
    print(Fore.YELLOW + f"🎵 Generating music for: {prompt}")
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

def play_audio(file_path):
    """Play the final audio file."""
    print(Fore.MAGENTA + f"\n▶ Playing the final generated music...")
    if os.name == "posix":
        os.system(f"afplay {file_path}")
    else:
        os.system(f"start {file_path}")

def dynamic_emotion_music_transition(start_emotion):
    """Handle the entire transition from start to positive end."""
    sequence = dynamic_transition_path(start_emotion)
    print(Fore.CYAN + f"\n✨ Emotion Transition Path: {Fore.WHITE}{' -> '.join(sequence)}\n")

    final_music_file = "dynamic_emotion_music.wav"

    for idx, emotion in enumerate(sequence):
        prompt = emotion_to_music.get(emotion.lower(), "A calm, neutral instrumental")
        chunk_file = generate_music(prompt, chunk_name=f"chunk_{idx}.wav", duration_tokens=256)
        final_music_file = append_music(final_music_file, chunk_file)
        print(Fore.GREEN + f"✅ Appended chunk {idx + 1}/{len(sequence)} for emotion: {emotion.capitalize()}")

    print(Fore.CYAN + f"\n🎉 Final dynamic music created: {final_music_file}")
    play_audio(final_music_file)

def spotify_recommendation_flow(emotion):
    """Handle the Spotify recommendation flow."""
    # First, initialize Spotify API
    sp = initialize_spotify()
    if not sp:
        print(Fore.RED + "Cannot continue without Spotify API connection")
        return
    
    # Ask for preferences
    shift_mood = input(Fore.CYAN + f"Your detected emotion is {emotion}. Do you want to shift your mood? (yes/no): ").strip().lower()
    artist_name = input(Fore.CYAN + "Enter preferred artist (or leave blank): ").strip()
    language_pref = input(Fore.CYAN + "Enter preferred language (or leave blank): ").strip()
    
    # Determine target mood
    if shift_mood == "yes" and emotion in EMOTION_SHIFT:
        target_mood = EMOTION_SHIFT[emotion]
        print(Fore.GREEN + f"Shifting mood from {emotion} to {target_mood}")
    else:
        target_mood = emotion
        print(Fore.GREEN + f"Finding songs for your current mood: {emotion}")
    
    # Get recommendations
    final_playlist = []
    
    # If artist is specified, get artist tracks with mood and language
    if artist_name:
        print(Fore.YELLOW + f"Finding {target_mood} songs by {artist_name}" + 
              (f" in {language_pref}" if language_pref else "") + "...")
              
        final_playlist = search_artist_tracks(sp, artist_name, 
                                          mood=target_mood, 
                                          language=language_pref)
        
        # If no results, try using mood-based search
        if not final_playlist:
            print(Fore.YELLOW + f"No tracks found for artist '{artist_name}', using mood recommendations instead")
            final_playlist = search_mood_tracks(sp, target_mood, language=language_pref)
    else:
        # Use mood-based search with language
        print(Fore.YELLOW + f"Finding {target_mood} songs" + 
              (f" in {language_pref}" if language_pref else "") + "...")
              
        final_playlist = search_mood_tracks(sp, target_mood, language=language_pref)
        
        # Try without language if no results
        if not final_playlist and language_pref:
            print(Fore.YELLOW + f"No {target_mood} tracks found in {language_pref}, trying without language filter...")
            final_playlist = search_mood_tracks(sp, target_mood)
    
    # Display results
    if final_playlist:
        print(Fore.GREEN + "\n🎵 Your Recommended Songs:")
        for i, track in enumerate(final_playlist, 1):
            source = track.get('source', 'unknown')
            print(Fore.CYAN + f"{i}. {track['name']} by {track['artists']} (source: {source})")
            print(Fore.WHITE + f"   Listen: {track['url']}")
            if track['preview']:
                print(Fore.WHITE + f"   Preview: {track['preview']}")
            print()
    else:
        print(Fore.RED + "\n❌ Sorry, no songs found matching your criteria.")
        print(Fore.YELLOW + "Try again with a different artist, mood, or language.")

def music_generation_flow(emotion):
    """Handle the music generation flow."""
    global music_gen_model, music_gen_processor
    
    # Load the model if not already loaded
    if music_gen_model is None or music_gen_processor is None:
        load_music_gen_model()
    
    # Generate dynamic emotion music
    print(Fore.BLUE + f"\n🎵 Starting Music Generation for {emotion.capitalize()} Emotion")
    dynamic_emotion_music_transition(emotion)

def main():
    """Main application flow."""
    print(Fore.BLUE + "🎼 Welcome to Emotion-Driven Music Tool! 🎼")
    print(Fore.YELLOW + "This application can detect your emotion and either recommend songs or generate music based on it.")
    
    # First, detect emotion
    emotion = detect_emotion()
    
    # Ask user what they want to do
    print(Fore.CYAN + "\nWhat would you like to do with your detected emotion?")
    print(Fore.WHITE + "1. Get music recommendations from Spotify")
    print(Fore.WHITE + "2. Generate custom music")
    
    choice = input(Fore.CYAN + "Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        spotify_recommendation_flow(emotion)
    elif choice == "2":
        music_generation_flow(emotion)
    else:
        print(Fore.RED + "Invalid choice. Please restart the application.")

if __name__ == "__main__":
    main()
