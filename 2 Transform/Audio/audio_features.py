import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 22050
DURATION = 10  # seconds per chunk
HOP_LENGTH = 512
N_MELS = 60


def extract_features(y, sr):
    """Extract (60, 431, 6) features with aligned dimensions"""
    stft = librosa.stft(y, n_fft=2048, hop_length=HOP_LENGTH)

    # 1. Mel Spectrogram (60, 431)
    mel = librosa.feature.melspectrogram(S=np.abs(stft), sr=sr, n_mels=N_MELS)

    # 2. MFCCs (60, 431)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=60)

    # 3. Chroma (12 → 60, 431)
    chroma = librosa.feature.chroma_stft(S=np.abs(stft), sr=sr)
    chroma = librosa.util.fix_length(chroma, size=mel.shape[1], axis=1)
    chroma = np.repeat(chroma, repeats=5, axis=0)  # 12×5=60

    # 4. Spectral Contrast (7 → 60, 431)
    contrast = librosa.feature.spectral_contrast(S=np.abs(stft), sr=sr, n_bands=6)
    contrast = librosa.util.fix_length(contrast, size=mel.shape[1], axis=1)
    # contrast = np.repeat(contrast, repeats=8, axis=0)[:60]  # 7×8=56 → pad to 60
    contrast = np.repeat(contrast, repeats=8, axis=0)  # 7×8=56
    pad_rows = 60 - contrast.shape[0]  # 60 - 56 = 4
    contrast = np.vstack([
        contrast,
        np.zeros((pad_rows, contrast.shape[1]))  # Add zero rows
    ])

    # 5. Tonnetz (6 → 60, 431)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz = librosa.util.fix_length(tonnetz, size=mel.shape[1], axis=1)
    tonnetz = np.repeat(tonnetz, repeats=10, axis=0)  # 6×10=60

    # # 6. Spectral Centroid (1 → 60, 431)
    # centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # centroid = librosa.util.fix_length(centroid, size=mel.shape[1], axis=1)
    # centroid = np.repeat(centroid, repeats=60, axis=0)

    # 6. log-mel
    centroid = librosa.power_to_db(mel, ref=np.max)

    # print(mel.shape)
    # print(mfcc.shape)
    # print(chroma.shape)
    # print(contrast.shape)
    # print(tonnetz.shape)
    # print(centroid.shape)
    # Stack features (60, 431, 6)
    return np.dstack([mel, mfcc, chroma, contrast, tonnetz, centroid])


# def main():
#     mood_df = pd.read_csv("moody_lyrics.csv")
#     mood_to_id = {mood: idx for idx, mood in enumerate(mood_df["Mood"].unique())}
#
#     total_chunks = 2000 * 3
#     features_tensor = torch.empty((total_chunks, 60, 431, 6), dtype=torch.float32)
#     mood_labels = torch.empty((total_chunks,), dtype=torch.long)
#
#     chunk_counter = 0
#     processed_files = 0
#
#     # Create progress bar
#     with tqdm(total=2000, desc="Processing Songs") as pbar:
#         for file in os.listdir("Music"):
#             if not file.endswith(".mp3"):
#                 continue
#
#             try:
#                 # Print current song
#                 song_name = os.path.splitext(file)[0]
#                 pbar.set_postfix_str(f"Analyzing: {song_name}")
#
#                 # Extract metadata
#                 artist, title = song_name.split(" - ")
#
#                 # Get mood
#                 match = mood_df[(mood_df["Artist"] == artist) &
#                                 (mood_df["Title"] == title)]
#                 if match.empty:
#                     pbar.write(f"Skipping {song_name} - no mood classification")
#                     continue
#
#                 mood = match["Mood"].values[0]
#                 mood_id = mood_to_id[mood]
#
#                 # Load audio
#                 y, sr = librosa.load(os.path.join("Music", file), sr=SAMPLE_RATE)
#
#                 # Process chunks
#                 for i in range(3):
#                     start = i * DURATION * sr
#                     end = start + DURATION * sr
#                     chunk = y[start:end]
#
#                     features = extract_features(chunk, sr)
#                     if features.shape != (60, 431, 6):
#                         pbar.write(f"Bad features in {song_name} chunk {i}")
#                         continue
#
#                     features_tensor[chunk_counter] = torch.from_numpy(features).float()
#                     mood_labels[chunk_counter] = mood_id
#                     chunk_counter += 1
#
#                 processed_files += 1
#                 pbar.update(1)
#
#             except Exception as e:
#                 pbar.write(f"Error processing {file}: {str(e)}")
#                 continue
#
#     # Final cleanup
#     features_tensor = features_tensor[:chunk_counter]
#     mood_labels = mood_labels[:chunk_counter]
#
#     print(f"\nSuccessfully processed {processed_files}/2000 songs")
#     print(f"Total valid chunks: {chunk_counter}")
#
#     torch.save(features_tensor, "combined_features.pt")
#     torch.save(mood_labels, "combined_labels.pt")
#
#     # Save mood mapping
#     pd.DataFrame(list(mood_to_id.items()), columns=["Mood", "Label"]).to_csv("mood_mapping.csv", index=False)

def main():
    mood_df = pd.read_csv("moody_lyrics.csv")
    mood_to_id = {mood: idx for idx, mood in enumerate(mood_df["Mood"].unique())}

    all_features = []
    all_mood_labels = []
    processed_files = 0

    # Create progress bar
    with tqdm(total=2000, desc="Processing Songs") as pbar:
        for file in os.listdir("music"):
            if not file.endswith(".mp3"):
                continue

            try:
                # Print current song
                song_name = os.path.splitext(file)[0]
                pbar.set_postfix_str(f"Analyzing: {song_name}")

                # Extract metadata
                artist, title = song_name.split(" - ")

                # Get mood
                match = mood_df[(mood_df["Artist"] == artist) &
                                (mood_df["Title"] == title)]
                if match.empty:
                    pbar.write(f"Skipping {song_name} - no mood classification")
                    continue

                mood = match["Mood"].values[0]
                mood_id = mood_to_id[mood]

                # Load audio
                y, sr = librosa.load(os.path.join("music", file), sr=SAMPLE_RATE)

                # Calculate chunks with padding
                total_samples = len(y)
                samples_per_chunk = DURATION * SAMPLE_RATE
                num_full_chunks = total_samples // samples_per_chunk
                remaining_samples = total_samples % samples_per_chunk
                remaining_seconds = remaining_samples / SAMPLE_RATE

                # Determine valid chunks
                num_chunks = num_full_chunks
                if remaining_seconds >= 9:
                    num_chunks += 1
                # print(num_chunks)
                # Process chunks
                for i in range(num_chunks):
                    start = i * samples_per_chunk
                    end = start + samples_per_chunk

                    # Handle last chunk padding
                    if i == num_chunks - 1 and remaining_samples >= 9 * SAMPLE_RATE:
                        chunk = y[start:]
                        chunk = librosa.util.fix_length(chunk, size=samples_per_chunk)
                    else:
                        chunk = y[start:end]

                    features = extract_features(chunk, sr)
                    if features.shape != (60, 431, 6):
                        pbar.write(f"Bad features in {song_name} chunk {i}")
                        continue

                    all_features.append(torch.from_numpy(features).float())
                    all_mood_labels.append(mood_id)

                processed_files += 1
                pbar.update(1)

            except Exception as e:
                pbar.write(f"Error processing {file}: {str(e)}")
                continue

    # Convert lists to tensors
    features_tensor = torch.stack(all_features)
    mood_labels = torch.tensor(all_mood_labels, dtype=torch.long)

    print(f"\nSuccessfully processed {processed_files}/2000 songs")
    print(f"Total valid chunks: {len(features_tensor)}")

    torch.save(features_tensor, "combined_features.pt")
    torch.save(mood_labels, "combined_labels.pt")
    pd.DataFrame(list(mood_to_id.items()), columns=["Mood", "Label"]).to_csv("mood_mapping.csv", index=False)
if __name__ == "__main__":
    main()