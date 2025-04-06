import pandas as pd

# Load datasets
moody_lyrics = pd.read_csv("./../../Raw Data/MoodyLyrics/ml_balanced.csv")
moody_4Q_lyrics = pd.read_csv("./../../Raw Data/MoodyLyrics/MoodyLyrics4Q.csv")

# Standardize column names
moody_lyrics.columns = moody_lyrics.columns.str.strip().str.lower()
moody_4Q_lyrics.columns = moody_4Q_lyrics.columns.str.strip().str.lower()

# Standardize strings for merging
for df in [moody_lyrics, moody_4Q_lyrics]:
    df['artist'] = df['artist'].str.strip()
    df['title'] = df['title'].str.strip()

# Add dummy mood column to moody_4Q_lyrics (if needed)
if 'mood' not in moody_4Q_lyrics.columns:
    moody_4Q_lyrics['mood'] = pd.NA

# Combine the two datasets
combined_df = pd.concat([
    moody_lyrics[['artist', 'title', 'mood']],
    moody_4Q_lyrics[['artist', 'title', 'mood']]
], ignore_index=True)

# Drop duplicates based on artist & title, keeping first non-null mood
combined_df = combined_df.sort_values(by='mood', na_position='last') \
                         .drop_duplicates(subset=['artist', 'title'], keep='first')

combined_df.columns = ['Artist', 'Title', 'Mood']
combined_df.to_csv("./Output/songs.csv")
