import pandas as pd
import requests
import os

# Get Deezer track URL based on song title and artist
def get_deezer_track_url(artist, title):
    query = f"{artist} {title}"
    url = f"https://api.deezer.com/search?q={query}"
    response = requests.get(url, timeout=10)  # timeout if taking too long
    data = response.json()

    # Check if we got results
    if data['total'] > 0:
        track_url = data['data'][0]['preview']
        return track_url
    else:
        return None


# Download MP3 from URL and save to 'music' folder
def download_mp3_from_url(url, artist, title):
    # Create 'music' folder if it doesn't exist
    if not os.path.exists("music"):
        os.makedirs("music")

    response = requests.get(url)
    file_path = f"music/{artist} - {title}.mp3"

    with open(file_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {artist} - {title}")


# Process csv file
def process_mood_lyrics_csv(input_csv):
    df = pd.read_csv(input_csv)

    for _, row in df.iterrows():
        artist = row['Artist']
        title = row['Title']

        print(f"Processing: {artist} - {title}")

        # Get the Deezer track preview URL
        track_url = get_deezer_track_url(artist, title)

        if track_url:
            try:
                download_mp3_from_url(track_url, artist, title)
            except Exception as e:
                print(f"Error downloading {artist} - {title}: {e}")
        else:
            print(f"Track not found: {artist} - {title}")


# Example usage
input_csv = 'moody_lyrics.csv'

process_mood_lyrics_csv(input_csv)
