from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Load data
X_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/x_train.csv")
X_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/x_test.csv")
y_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/y_train.csv")
y_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/y_test.csv")

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Hyperparameters
vocab_size = 10000
embedding_dim = 128
max_length = 500

# Architecture for Model T₁' (no LSTM layer)
model_T1_prime = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Flatten(),  # Flatten the output to make it compatible with Dense layers
    Dense(128, activation='relu'),  # First fully connected layer
    Dropout(0.2),  # Optional dropout to prevent overfitting
    Dense(128, activation='relu'),  # Second fully connected layer
    Dense(4, activation='softmax')  # Output layer with 4 classes
])
model_T1_prime.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_T1_prime.summary()

# Train the model
model_T1_prime.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)

# Evaluate model on test set
loss, accuracy = model_T1_prime.evaluate(X_test, y_test_cat, verbose=1)

print(f"Test Accuracy for Model T1': {accuracy * 100:.2f}%")
