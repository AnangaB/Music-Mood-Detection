import pandas as pd
import pandas as pd
import re

# Load the dataset
df = pd.read_csv("./../../../1 Extract/Data/Lyrics/2 Make Mood Lyrics Dataset/Output/songs_lyrics_mood.csv")

# Function to clean lyrics
def clean_lyrics(lyrics):
    # Split lyrics into lines and remove empty lines or lines containing just smt of form [ .* ]
    lines = [line for line in lyrics.split('\n') if line.strip() != '' and not re.match(r'^\[.*\]$', line.strip())]
    return '\n'.join(lines)

df["lyrics"] = df["lyrics"].map(clean_lyrics)


df.to_csv("./Output/Clean_lyrics_mood_dataset.csv")