# Testing A1 model for binary classification
import torch
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load preprocessed features and labels
features_tensor = torch.load("combined_features.pt")  # shape: (5663, 60, 431, 6)
labels_tensor = torch.load("combined_labels.pt")  # shape: (5663,)

# Convert to NumPy for compatibility with TensorFlow
X = features_tensor.numpy()
y = labels_tensor.numpy()

# Convert to binary labels (0: negative, 1: positive)
# angry(1) and sad(3) -> 0, happy(2) and relaxed(0) -> 1
y_binary = np.where((y == 1) | (y == 3), 0, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary  # Maintain class balance
)


# Build the CNN model for binary classification
def build_cnn_model(input_shape):
    model = models.Sequential()
    # Conv-Pool Block 1
    model.add(layers.Conv2D(128, (6, 6), activation='relu',
                            input_shape=input_shape,
                            kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv-Pool Block 2
    model.add(layers.Conv2D(256, (5, 5), activation='relu',
                            padding='same',
                            kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Conv-Pool Block 3
    model.add(layers.Conv2D(512, (4, 4), activation='relu',
                            padding='same',
                            kernel_regularizer=l2(0.00001)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary output

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Model input shape (60, 431, 6)
input_shape = (60, 431, 6)
model = build_cnn_model(input_shape)

# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")