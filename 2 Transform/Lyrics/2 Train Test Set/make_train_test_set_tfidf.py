import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from helper import preprocess_text
# Load the cleaned-up lyrics dataset
df = pd.read_csv("./../1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")

# Preprocess all lyrics
lyrics = df["lyrics"].astype(str).apply(preprocess_text).values

# Label encoding for mood (if not already encoded)
labels = df["Mood"]
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Using top 10,000 words
X_tfidf = tfidf_vectorizer.fit_transform(lyrics).toarray()  # Convert the TF-IDF matrix to an array

# Dimensionality Reduction using PCA
pca = PCA(n_components=128)  # Reduce to 128 dimensions
X_reduced = pca.fit_transform(X_tfidf)  # Apply PCA to the TF-IDF vectors

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, labels_encoded, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Output results to CSV
output_folder = r"Output/TF-IDF"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)

