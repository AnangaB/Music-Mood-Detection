import pandas as pd
from sklearn.model_selection import train_test_split
from helper import preprocess_text  
from sklearn.preprocessing import LabelEncoder

# Load the cleaned-up lyrics dataset
df = pd.read_csv("./../1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")

# Preprocess all lyrics
lyrics = df["lyrics"].astype(str).apply(preprocess_text).values

# Label encoding for mood
labels = df["Mood"]
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lyrics, labels_encoded, test_size=0.2, random_state=42)



print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Output results to CSV
output_folder = r"Output\BERT"
pd.DataFrame(X_train).to_csv(f"{output_folder}/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(f"{output_folder}/X_test.csv", index=False)
pd.Series(y_train).to_csv(f"{output_folder}/y_train.csv", index=False)
pd.Series(y_test).to_csv(f"{output_folder}/y_test.csv", index=False)

