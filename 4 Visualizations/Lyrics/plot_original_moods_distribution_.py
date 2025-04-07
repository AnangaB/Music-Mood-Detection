import pandas as pd
import matplotlib.pyplot as plt
import seaborn

seaborn.set()


# Load datasets with corrected paths
original_dataset = pd.read_csv(r"./../../1 Extract/Data/Lyrics/3 Make Mood Lyrics Dataset/Output/songs_lyrics_mood.csv")
dataset_post_split = pd.read_csv(r"./../../2 Transform/Lyrics/1 Clean Up Lyrics/Output/Clean_lyrics_mood_dataset.csv")

# Get value counts as percentages
original_counts = original_dataset['Mood'].value_counts(normalize=True) * 100
post_split_counts = dataset_post_split['Mood'].value_counts(normalize=True) * 100

# Plot side-by-side bar chart
fig, ax = plt.subplots(figsize=(10, 6))
all_moods = sorted(set(original_counts.index).union(set(post_split_counts.index)))
x = range(len(all_moods))
bar_width = 0.35

ax.bar([i - bar_width/2 for i in x], [original_counts.get(mood, 0) for mood in all_moods],
       width=bar_width, label='Original')
ax.bar([i + bar_width/2 for i in x], [post_split_counts.get(mood, 0) for mood in all_moods],
       width=bar_width, label='Post-split')

# Styling
ax.set_xlabel("Mood")
ax.set_ylabel("Percentage (%)")
ax.set_title("Mood Distribution: Original vs. After Splitting Up Lyrics")
ax.set_xticks(x)
ax.set_xticklabels(all_moods, rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("Output/mood_dist.png")
