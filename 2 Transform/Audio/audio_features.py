import pandas as pd
import librosa
import numpy as np
import os
import torch
from scipy import interpolate


# Function to get audio features from a local MP3 file
def get_audio_features_from_file(file_path):
    # Load audio with librosa
    y, sr = librosa.load(file_path, sr=None)

    # Calculate the feature matrices
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)  # Changed to 128 to match Mel spectrogram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Calculate the Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)

    # Calculate the log-Mel spectrogram
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Function to resample features to the target number of time frames (60)
    def resample_feature(feature_matrix, target_frames=60):
        num_frames = feature_matrix.shape[1]
        if num_frames == target_frames:
            return feature_matrix  # Already the desired length
        elif num_frames < target_frames:
            # If the number of frames is less than the target, just repeat the data (pad)
            return np.pad(feature_matrix, ((0, 0), (0, target_frames - num_frames)), mode='edge')
        else:
            # Interpolate to target number of frames (60)
            time = np.arange(num_frames)
            resampled_time = np.linspace(0, num_frames - 1, target_frames)
            resampled_feature = np.zeros((feature_matrix.shape[0], target_frames))
            for i in range(feature_matrix.shape[0]):
                resampled_feature[i] = np.interp(resampled_time, time, feature_matrix[i])
            return resampled_feature

    # Resample each feature to have 60 time points
    mfcc_resampled = resample_feature(mfcc)
    chroma_resampled = resample_feature(chroma)
    tonnetz_resampled = resample_feature(tonnetz)
    contrast_resampled = resample_feature(contrast)
    mel_resampled = resample_feature(mel)
    log_mel_resampled = resample_feature(log_mel)

    # Return each feature matrix separately
    return {
        "mfcc": mfcc_resampled,
        "chroma": chroma_resampled,
        "tonnetz": tonnetz_resampled,
        "contrast": contrast_resampled,
        "mel": mel_resampled,
        "log_mel": log_mel_resampled
    }


# Function to process the MoodyLyrics CSV and save features to a PyTorch tensor
def process_mood_lyrics_csv(input_csv, music_folder, output_tensor):
    df = pd.read_csv(input_csv)

    all_features = []

    for _, row in df.iterrows():
        artist = row['Artist']
        title = row['Title']
        mood = row['Mood']

        print(f"Processing: {artist} - {title}")

        # Construct the file path based on the artist and title
        file_name = f"{artist} - {title}.mp3"
        file_path = os.path.join(music_folder, file_name)

        if os.path.exists(file_path):
            try:
                features = get_audio_features_from_file(file_path)

                # Concatenate the features to form a single array
                # Shape for each feature: (num_features, 60)
                features_combined = np.concatenate([
                    features["mfcc"],  # Shape: (128, 60)
                    features["chroma"],  # Shape: (12, 60)
                    features["tonnetz"],  # Shape: (6, 60)
                    features["contrast"],  # Shape: (7, 60)
                    features["mel"],  # Shape: (128, 60)
                    features["log_mel"]  # Shape: (128, 60)
                ], axis=0)  # Now shape: (409, 60)

                # Append the features for each song
                all_features.append(features_combined)
            except Exception as e:
                print(f"Error processing {artist} - {title}: {e}")
                # Handle error case by adding zero-padded features
                zero_filled_features = np.zeros((409, 60))  # Zero-padded feature matrix
                all_features.append(zero_filled_features)
        else:
            print(f"File not found: {file_name}")
            # Handle missing file case by adding zero-padded features
            zero_filled_features = np.zeros((409, 60))  # Zero-padded feature matrix
            all_features.append(zero_filled_features)

    # Convert the list into a 3D NumPy array (num_songs, 409, 60)
    all_features_array = np.array(all_features)

    # Debugging: check the shape of the final combined array before conversion to tensor
    print(f"Shape of final array: {all_features_array.shape}")

    # Now reshape the array to have shape (num_songs, 60, 409, 6)
    # The last dimension (6) is the number of feature types
    all_features_array_reshaped = all_features_array[:, :, :, np.newaxis]  # Add a new dimension for features
    all_features_array_reshaped = np.repeat(all_features_array_reshaped, 6,
                                            axis=-1)  # Repeat along the feature dimension

    # Convert the NumPy array to a PyTorch tensor
    features_tensor = torch.tensor(all_features_array_reshaped, dtype=torch.float32)

    # Save the tensor to a .pt file
    torch.save(features_tensor, output_tensor)
    print(f"Results saved to {output_tensor}")


# Example usage
input_csv = 'moody_lyrics.csv'  # Path to your MoodyLyrics CSV file
music_folder = 'music'  # Path to your music folder where the MP3 files are stored
output_tensor = 'moody_lyrics_with_audio_features.pt'  # Path to save the PyTorch tensor with features

process_mood_lyrics_csv(input_csv, music_folder, output_tensor)
