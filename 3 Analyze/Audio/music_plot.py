import torch
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the labels
labels = torch.load("combined_labels.pt")
if isinstance(labels, torch.Tensor):
    labels = labels.numpy()

# Load mood mapping
mood_map = pd.read_csv("mood_mapping.csv")
label_to_mood = dict(zip(mood_map['Label'], mood_map['Mood']))

# Count class occurrences
label_counts = Counter(labels)
classes = sorted(label_counts.keys())
counts = [label_counts[cls] for cls in classes]
class_names = [label_to_mood[cls].capitalize() for cls in classes]  # Get mood names

# Create professional plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
palette = sns.color_palette("husl", len(classes))

# Bar plot with mood names
ax = sns.barplot(x=class_names, y=counts, palette=palette,
                 edgecolor="black", linewidth=1.2)

# Customize plot
plt.title("Class Distribution in MoodyLyrics Dataset", fontsize=14, pad=20)
plt.xlabel("Mood Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.xticks(rotation=45)

# Add annotations
total = sum(counts)
for i, (count, mood) in enumerate(zip(counts, class_names)):
    # Count above bars
    ax.text(i, count + max(counts) * 0.02, f"{count}",
            ha='center', va='bottom', fontsize=10)

    # Percentage inside bars (white text)
    percentage = 100 * count / total
    ax.text(i, count // 2, f"{percentage:.1f}%",
            ha='center', va='center',
            color="white", fontweight='bold')

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()


# A different plot
# Load training history
history = pd.read_csv('training_history_A1.csv')

# Create figure with two subplots
plt.figure(figsize=(7, 5))
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelpad'] = 12

# Plot accuracy curves
# plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training', color='#3498db', linewidth=2)
plt.plot(history['val_accuracy'], label='Validation', color='#e74c3c', linewidth=2)
plt.title('A1 Model Accuracy', pad=20, fontweight='bold')
plt.ylabel('Accuracy', labelpad=10)
plt.xlabel('Epoch', labelpad=10)
plt.ylim(0.2, 0.3)
plt.legend(framealpha=1)

# Plot loss curves with log scale
# plt.subplot(1, 2, 2)
# plt.semilogy(history['loss'], label='Training', marker='o', color='#3498db', linewidth=2)
# plt.semilogy(history['val_loss'], label='Validation', marker='s', color='#e74c3c', linewidth=2)
# plt.title('Model Loss', pad=20, fontweight='bold')
# plt.ylabel('Loss (log scale)', labelpad=10)
# plt.xlabel('Epoch', labelpad=10)
# plt.legend(framealpha=1)

# Adjust layout and save
plt.tight_layout()
plt.savefig('training_curves_A1.png', dpi=300, bbox_inches='tight')
plt.show()

# Print training summary
print("\nTraining Summary:")
print(f"Final Training Accuracy: {history['accuracy'].iloc[-1]:.4f}")
print(f"Final Validation Accuracy: {history['val_accuracy'].iloc[-1]:.4f}")
print(f"Final Training Loss: {history['loss'].iloc[-1]:.4f}")
print(f"Final Validation Loss: {history['val_loss'].iloc[-1]:.4f}")