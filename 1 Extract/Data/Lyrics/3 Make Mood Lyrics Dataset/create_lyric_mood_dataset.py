import pandas as pd

lyrics_data_set = pd.read_csv("./../2 Lyrics Extraction/Lyrics Data Output/lyrics_output.csv")
mood_data_set = pd.read_csv("./../1 Compile Songs List/Output/songs.csv")

output_file = "./Output/songs_lyrics_mood.csv"

training_set = lyrics_data_set.merge(
    mood_data_set,
    left_on=["title", "artist"],
    right_on=["Title", "Artist"]
)[["Title", "Artist","lyrics","Mood"]].drop_duplicates(subset=['Title', 'Artist'])
training_set.to_csv(output_file, index=False)


# Check for missing or empty lyrics
missing_lyrics = training_set['lyrics'].isna() | (training_set['lyrics'].str.strip() == "")
num_missing_lyrics = missing_lyrics.sum()
print(f"Number of entries with missing or empty lyrics: {num_missing_lyrics}")

num_unique_songs = training_set[['Title', 'Artist']].shape[0]
print(f"Number of songs: {num_unique_songs}")
