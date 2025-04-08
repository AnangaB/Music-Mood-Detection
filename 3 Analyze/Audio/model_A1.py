import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load preprocessed features and labels
features_tensor = torch.load("combined_features.pt")  # shape: (5663, 60, 431, 6)
labels_tensor = torch.load("combined_labels.pt")      # shape: (5663,)

# Convert to NumPy for compatibility with TensorFlow
X = features_tensor.numpy()
y = labels_tensor.numpy()

# Encode labels if they are strings
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode the labels
y_one_hot = to_categorical(y_encoded, num_classes=len(np.unique(y_encoded)))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_encoded)

# Build the CNN model
def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (6, 6), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (4, 4), activation='relu', padding='same', kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(y_one_hot.shape[1], activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Model input shape (60, 431, 6)
input_shape = (60, 431, 6)
model = build_cnn_model(input_shape)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
