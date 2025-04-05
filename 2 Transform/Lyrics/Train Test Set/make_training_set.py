import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your dataset
df = pd.read_csv("./../../../1 Extract/Data/Lyrics/2 Make Mood Lyrics Dataset/Output/songs_lyrics_mood.csv")  # Contains lyrics, Title, Artist, Mood

lyrics = df["lyrics"].astype(str).values
labels = df["Mood"]

# Encode labels (assuming 4 emotion classes)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize lyrics
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

# Pad sequences
max_length = 500
padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(padded, labels_encoded,train_size=.8, test_size=0.2, random_state=42)

output_folder = "Output"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)