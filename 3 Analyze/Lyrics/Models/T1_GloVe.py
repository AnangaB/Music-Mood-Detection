import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from helper import get_GloVe_train_test_data 

X_train, X_test, y_train, y_test = get_GloVe_train_test_data()

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

# Model architecture with LSTM
model_T1 = Sequential([
    # LSTM layer expects 3D input: (batch_size, timesteps, features)
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], 1)),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # Output layer for 4 classes
])

# Compile the model
model_T1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model_T1.summary()

# Reshape input for LSTM: Convert 2D input (samples, features) to 3D input (samples, timesteps, features)
# Since we have sentence embeddings of size 100, reshape X_train and X_test
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # (samples, embedding_size, 1)
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # (samples, embedding_size, 1)

# Train the model
model_T1.fit(X_train_reshaped, y_train_cat, validation_data=(X_test_reshaped, y_test_cat), epochs=20, batch_size=512)

# Evaluate the model on test set
loss, accuracy = model_T1.evaluate(X_test_reshaped, y_test_cat, verbose=1)

# Output the test accuracy
print(f"Test Accuracy for Model T1: {accuracy * 100:.2f}%")
