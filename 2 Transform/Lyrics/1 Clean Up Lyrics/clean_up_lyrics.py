import pandas as pd
import re

# Load the dataset
df = pd.read_csv("./../../../1 Extract/Data/Lyrics/3 Make Mood Lyrics Dataset/Output/songs_lyrics_mood.csv")

# Function to clean lyrics
def clean_lyrics(lyrics):
    # Split lyrics into lines and remove empty lines or lines containing just something of form [ .* ]
    lines = [line for line in lyrics.split('\n') if line.strip() != '' and not re.match(r'^\[.*\]$', line.strip())]
    return '\n'.join(lines)

df["lyrics"] = df["lyrics"].map(clean_lyrics)

labels_encoded = df["Mood"].tolist()

# Function to split lyrics into subtexts of 4 lines
def split_lyrics(lyrics, labels_encoded, lines_per_subtext=4):
    subtexts = []
    expanded_labels = []  # List to store the repeated labels for subtexts
    for i, lyric in enumerate(lyrics):
        lines = lyric.split('\n')  # No need to check for empty lines again
        num_subtexts = (len(lines) + lines_per_subtext - 1) // lines_per_subtext  # Calculate the number of subtexts

        # Repeat the label for each subtext
        expanded_labels.extend([labels_encoded[i]] * num_subtexts)
        for j in range(0, len(lines), lines_per_subtext):
            subtext = '\n'.join(lines[j:j + lines_per_subtext])  
            subtexts.append(subtext)
    
    return subtexts, expanded_labels

# Split lyrics into subtexts of 4 lines
split_lyrics_list, expanded_labels = split_lyrics(df["lyrics"], labels_encoded, lines_per_subtext=4)

# Create a new DataFrame from the split lyrics and the expanded labels
new_df = pd.DataFrame(zip(split_lyrics_list, expanded_labels), columns=["lyrics", "Mood"])
# Drop rows where lyrics are NaN or empty/whitespace only
new_df = new_df.dropna(subset=['lyrics'])
new_df = new_df[new_df['lyrics'].str.strip() != '']

# Save the cleaned dataset with subtexts
new_df.to_csv("./Output/Clean_lyrics_mood_dataset.csv", index=False)


# Print summary
print(f"Original number of songs: {len(df)}")
print(f"Number of subtext rows after splitting: {len(new_df)}")

print("Distribution of moods in original dataset: ", df["Mood"].value_counts())
print("Distribution of moods in after splitting lyrics dataset: ", new_df["Mood"].value_counts())