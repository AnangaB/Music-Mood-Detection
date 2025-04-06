import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the root directory
root_dir = os.path.join("..", "..", "3 Analyze", "Lyrics", "Models", "Output")

# Map each file to its label using os.path.join
files = {
    os.path.join(root_dir, "T1_GloVe_training_history.csv"): "T1 GloVe",
    os.path.join(root_dir, "T1_Prime_BOW_training_history.csv"): "T1 Prime BOW",
    os.path.join(root_dir, "T1_Prime_TF_IDF_training_history.csv"): "T1 Prime TF-IDF",
    os.path.join(root_dir, "T1_Word2Vec_training_history.csv"): "T1 Word2Vec"
}

# Define a color palette (you can pick any colors you like)
color_map = {
    "T1 GloVe": "#1f77b4",       # Blue
    "T1 Prime BOW": "#2ca02c",   # Green
    "T1 Prime TF-IDF": "#9467bd",# Purple
    "T1 Word2Vec": "#ff7f0e"     # Orange
}

plt.figure(figsize=(12, 6))

# Plot training and validation accuracy with consistent colors
for file, label in files.items():
    df = pd.read_csv(file)
    color = color_map[label]
    plt.plot(df["epoch"], df["accuracy"], label=f"{label} - Train", linestyle='-', color=color)
    plt.plot(df["epoch"], df["val_accuracy"], label=f"{label} - Val", linestyle='--', color=color)

plt.title("Training & Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()

# Place legend outside the plot on the right
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Save and show the plot
plt.savefig("Output/all_models_training_accuracy.png", bbox_inches='tight')
plt.show()
