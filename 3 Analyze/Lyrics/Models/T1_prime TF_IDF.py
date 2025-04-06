from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
from helper import get_tf_idf_train_test_data 

X_train, X_test, y_train, y_test = get_tf_idf_train_test_data()

# One-hot encoding labels for categorical classification
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

vocab_size = 10000  
input_length = X_train.shape[1] 


# Architecture for Model T1'
model_T1_prime = Sequential([
    # First fully connected layer (hidden layer)
    Dense(128, activation='relu'),
    Dropout(0.2),  
    # Output layer with 4 classes (softmax for multi-class classification)
    Dense(4, activation='softmax')
])

# Compile model
model_T1_prime.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model_T1_prime.summary()

# Train the model
model_T1_prime.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=100, batch_size=256)

# Evaluate model on test set
loss, accuracy = model_T1_prime.evaluate(X_test, y_test_cat, verbose=1)

print(f"Test Accuracy for Model T1': {accuracy * 100:.2f}%")
