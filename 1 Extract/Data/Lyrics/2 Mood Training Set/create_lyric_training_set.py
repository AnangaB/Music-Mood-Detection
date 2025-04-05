import pandas as pd

lyrics_data_set = pd.read_csv("./../1 Lyrics Extraction/Lyrics Data Output/lyrics_output.csv")
mood_data_set = pd.read_csv("./../../Raw Data/songs.csv")

output_file = "./Output/songs_lyrics_mood.csv"

training_set = lyrics_data_set.merge(
    mood_data_set,
    left_on=["title", "artist"],
    right_on=["Title", "Artist"]
)[["Title", "Artist","lyrics","Mood"]]
training_set.to_csv(output_file)
