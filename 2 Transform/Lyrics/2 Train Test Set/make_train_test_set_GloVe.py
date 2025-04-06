import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from helper import preprocess_text

# Load the cleaned-up lyrics dataset
df = pd.read_csv("./../1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")

# Preprocess all lyrics
lyrics = df["lyrics"].astype(str).apply(preprocess_text).values

# Label encoding for mood
labels = df["Mood"]
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Tokenize the lyrics into words
tokenized_lyrics = [text.split() for text in lyrics]

# Load GloVe embeddings (100-dimensional vectors for this example)
def load_glove_embeddings(file_path):
    glove_embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    return glove_embeddings

# Load pretrained GloVe embeddings
glove_embeddings = load_glove_embeddings("./../../../1 Extract/Data/Raw Data/glove.6B.100d.txt")

# Function to create sentence vectors by averaging word vectors
def get_sentence_vector(tokens, embedding_dict, embedding_size=100):
    sentence_vector = np.zeros(embedding_size)
    count = 0
    for word in tokens:
        if word in embedding_dict:
            sentence_vector += embedding_dict[word]
            count += 1
    if count > 0:
        sentence_vector /= count
    return sentence_vector

# Create sentence vectors by averaging the GloVe word vectors in each sentence
sentence_vectors = np.array([get_sentence_vector(tokens, glove_embeddings) for tokens in tokenized_lyrics])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentence_vectors, labels_encoded, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Output results to CSV
output_folder = r"Output/GloVe"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)
