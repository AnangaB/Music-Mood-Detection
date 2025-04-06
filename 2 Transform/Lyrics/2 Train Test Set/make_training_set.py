import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv("./../1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")  # Contains lyrics, Title, Artist, Mood

lyrics = df["lyrics"].astype(str).values
labels = df["Mood"]

# Encode labels (assuming 4 emotion classes)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Function to split lyrics into subtexts of 4 lines
def split_lyrics(lyrics, lines_per_subtext=4):
    subtexts = []
    expanded_labels = []  # List to store the repeated labels for subtexts
    for i, lyric in enumerate(lyrics):
        lines = [line for line in lyric.split('\n') if line.strip() != '']  # Remove empty lines
        num_subtexts = (len(lines) + lines_per_subtext - 1) // lines_per_subtext  # Calculate the number of subtexts
        # Repeat the label for each subtext
        expanded_labels.extend([labels_encoded[i]] * num_subtexts)
        for j in range(0, len(lines), lines_per_subtext):
            # Join the next 'lines_per_subtext' lines (including the last one if fewer than 'lines_per_subtext' remain)
            subtext = ' '.join(lines[j:j + lines_per_subtext])  
            subtexts.append(subtext)
    return subtexts, expanded_labels

# Split lyrics into subtexts of 4 lines
split_lyrics_list, expanded_labels = split_lyrics(lyrics, lines_per_subtext=4)

# Tokenize lyrics (subtexts)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(split_lyrics_list)
sequences = tokenizer.texts_to_sequences(split_lyrics_list)

# Pad sequences
max_length = 500
padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, expanded_labels, test_size=0.2, random_state=42)

# Output results to CSV
output_folder = "Output"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)
