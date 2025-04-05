from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd

X_train = pd.read_csv("./../../../2 Transform/Lyrics/Train Test Set/Output/x_train.csv")
X_test = pd.read_csv("./../../../2 Transform/Lyrics/Train Test Set/Output/x_test.csv")
y_train = pd.read_csv("./../../../2 Transform/Lyrics/Train Test Set/Output/y_train.csv")
y_test = pd.read_csv("./../../../2 Transform/Lyrics/Train Test Set/Output/y_test.csv")

# Convert labels to one-hot
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

vocab_size = 10000
embedding_dim = 128
max_length = 500

model_T1 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(64, activation='tanh', recurrent_activation='sigmoid', dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')  # 4-class classification
])

model_T1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_T1.summary()


# model_T1_prime = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
#     Dense(128, activation='relu'),
#     Dense(4, activation='softmax')
# ])

# model_T1_prime.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_T1_prime.summary()

# Train T₁
model_T1.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)

# Evaluate model on test set
loss, accuracy = model_T1.evaluate(X_test, y_test_cat, verbose=1)

print(f"Test Accuracy for Model T₁: {accuracy * 100:.2f}%")


# Train T₁′
# model_T1_prime.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=5, batch_size=32)
