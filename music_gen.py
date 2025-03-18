import torch
import torchaudio
import os
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import time
import random
from colorama import Fore, Style, init
init(autoreset=True)

# Load the MusicGen model
print(Fore.CYAN + "🎶 Loading MusicGen model... (This may take time initially)")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
print(Fore.GREEN + "✅ Model loaded successfully!\n")

# Expanded Emotion-to-Music Prompt Mapping
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
    "content": "A smooth lo-fi chill beat with jazzy keys and subtle vinyl crackles."
}

positive_emotions = ["peaceful", "joyful", "content", "dreamy"]
all_emotions = list(emotion_to_music.keys())

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
    print(Fore.YELLOW + f"🎵 Generating music for: {prompt}")
    inputs = processor(text=prompt, return_tensors="pt")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=duration_tokens)
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

if __name__ == "__main__":
    print(Fore.BLUE + "🎼 Welcome to Dynamic Emotion-based Music Generator! 🎼")
    user_emotion = input(Fore.CYAN + "Enter your current emotion (happy, sad, angry, anxious, peaceful, energetic, romantic, mysterious, nostalgic, dreamy, joyful, content): ").strip().lower()

    if user_emotion not in emotion_to_music:
        print(Fore.RED + "⚠ Invalid emotion! Defaulting to 'neutral (peaceful)'...")
        user_emotion = "peaceful"

    start = time.time()
    dynamic_emotion_music_transition(user_emotion)
    end = time.time()
    print(Fore.GREEN + f"\n⏱ Total time taken: {round(end - start, 2)} seconds")
