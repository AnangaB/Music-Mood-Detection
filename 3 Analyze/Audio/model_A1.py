from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the tensor data
features_tensor = torch.load('moody_lyrics_with_audio_features.pt')

# Load the labels (Mood) from the moody_lyrics.csv file
moody_lyrics_df = pd.read_csv('test.csv') # 10 samples

# Extract the 'Mood' column from the DataFrame for y
y = moody_lyrics_df['Mood'].values

# Encode the 'Mood' labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Checking the unique labels and their encoding
print("Unique moods:", label_encoder.classes_)
print("Encoded moods:", y_encoded[:10])  # Show the first 10 encoded labels

# Now, you need to match the tensor size with the number of samples in y (the mood labels)
# The tensor is of shape [10, 409, 60, 6], we assume that the 409 corresponds to the number of samples.
# Ensure that `y_encoded` has the same number of samples as the first dimension of the tensor

# Checking the shape of features_tensor and length of y_encoded
print("Features tensor shape:", features_tensor.shape)
print("Encoded mood labels length:", len(y_encoded))

# Make sure both have the same number of samples
assert len(y_encoded) == features_tensor.shape[0], "Number of samples in features and labels do not match."

# Convert the tensor to a numpy array for splitting
X = features_tensor.numpy()

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Optional: Convert back to tensors if needed
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# One-hot encode the labels
y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)

# Define the CNN model
def build_cnn_model(input_shape):
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(128, (6, 6), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block
    model.add(layers.Conv2D(512, (4, 4), activation='relu', padding='same', kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the 3D tensor to 1D
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.6))

    # Fully Connected Layer 2 (output layer)
    model.add(layers.Dense(4, activation='softmax'))  # Assuming 4 classes for mood classification

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Example input shape (no resizing, keep original shape)
input_shape = (409, 60, 6)  # Using the original shape of the data

# Build the model
model = build_cnn_model(input_shape)

# Summary of the model architecture
# model.summary()

# Train the model with the original features (no resizing needed)
model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=2)

# Print the test accuracy and loss
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Optionally, you can make predictions using the test set
y_pred = model.predict(X_test)

# Convert predictions back to the label indices (for analysis or metrics like confusion matrix)
y_pred_classes = y_pred.argmax(axis=-1)  # Get the class with the highest probability

# Optionally, print out some predictions and their actual labels for inspection
print("Predicted Classes:", y_pred_classes[:10])
print("True Classes:", y_test[:10])
