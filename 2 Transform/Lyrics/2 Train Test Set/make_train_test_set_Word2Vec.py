import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np
from helper import preprocess_text

# Load the cleaned-up lyrics dataset
df = pd.read_csv("./../1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")

# Preprocess all lyrics
lyrics = df["lyrics"].astype(str).apply(preprocess_text).values

# Label encoding for mood
labels = df["Mood"]
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize the lyrics into words for Word2Vec
tokenized_lyrics = [text.split() for text in lyrics]

# Train Word2Vec model on tokenized lyrics
word2vec_model = Word2Vec(sentences=tokenized_lyrics, vector_size=128, window=5, min_count=1, workers=4)

# Function to create sentence vectors by averaging word vectors
def get_sentence_vector(tokens):
    # Initialize vector with zeros
    sentence_vector = np.zeros(128)
    count = 0
    for word in tokens:
        if word in word2vec_model.wv:
            sentence_vector += word2vec_model.wv[word]
            count += 1
    # Return the average of the word vectors for the sentence (if count > 0)
    if count > 0:
        sentence_vector /= count
    return sentence_vector

# Create sentence vectors by averaging the word vectors in each sentence
sentence_vectors = np.array([get_sentence_vector(tokens) for tokens in tokenized_lyrics])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, labels_encoded, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Output results to CSV
output_folder = r"Output/Word2Vec"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)
