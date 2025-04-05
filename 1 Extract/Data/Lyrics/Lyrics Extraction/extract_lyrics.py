import pandas as pd
import requests
import time
import os
# Load existing lyrics data if it exists
output_path = "./Lyrics Data Output/lyrics_output.csv"

if os.path.exists(output_path):
    lyrics_df = pd.read_csv(output_path)
    existing_songs = set(zip(lyrics_df['title'], lyrics_df['artist']))
else:
    lyrics_df = pd.DataFrame(columns=["title", "artist", "lyrics"])
    existing_songs = set()

# Load songs
songs_df = pd.read_csv("./../../Raw Data/songs.csv")  # has cols 'Title' and 'Artist'

# Load Spotify data (fallback source)
spotify_data = pd.read_csv("./../../Raw Data/Spotify Data/spotify_millsongdata.csv")

# Filter out songs that have already been processed
remaining_songs_df = songs_df[~songs_df.apply(lambda row: (row['Title'], row['Artist']) in existing_songs, axis=1)]

print(f"{len(remaining_songs_df)} songs need lyrics out of the total {len(songs_df)} songs.")

lyrics_data = []
for index, row in remaining_songs_df.iterrows():
    title = row['Title']
    artist = row['Artist']
    
    # Skip songs already in the output CSV
    if (title, artist) in existing_songs:
        print(f"Skipping already found lyrics for: {title} - {artist}")
        continue
    
    # Try fetching lyrics from the Lyrics API
    lyrics = ""
    url = f"https://api.lyrics.ovh/v1/{artist}/{title}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        lyrics = response.json().get("lyrics", "")
        
        if lyrics:
            print(f"Found lyrics from API: {title} - {artist}")
        else:
            print(f"Lyrics not found from API: {title} - {artist}")
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error for {title} - {artist}: {http_err}")
    except Exception as e:
        print(f"Error with {title} - {artist}: {e}")
    
    # If no lyrics were found from the Lyrics API, try to find from Spotify dataset
    if not lyrics:
        spotify_match = spotify_data[(spotify_data['song'].str.lower() == title.lower()) & 
                                 (spotify_data['artist'].str.lower() == artist.lower())]
        if not spotify_match.empty:
            lyrics = spotify_match.iloc[0]['text']  # Assuming the Spotify data has a 'Lyrics' column
            print(f"Found lyrics from Spotify dataset: {title} - {artist}")
        else:
            print(f"No lyrics found for {title} - {artist} in Spotify dataset either.")

    if lyrics:
        lyrics_data.append({
            "title": title,
            "artist": artist,
            "lyrics": lyrics
        })

    time.sleep(0.2)
# --- Save results ---
if lyrics_data:
    new_lyrics_df = pd.DataFrame(lyrics_data)
    lyrics_df = pd.concat([lyrics_df, new_lyrics_df], ignore_index=True)
    lyrics_df.to_csv(output_path, index=False)
