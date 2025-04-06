import pandas as pd


def get_bow_train_test_data():
    X_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/BoW/x_train.csv")
    X_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/BoW/x_test.csv")
    y_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/BoW/y_train.csv")
    y_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/BoW/y_test.csv")
    return X_train,X_test,y_train,y_test


def get_tf_idf_train_test_data():
    X_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/TF-IDF/x_train.csv")
    X_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/TF-IDF/x_test.csv")
    y_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/TF-IDF/y_train.csv")
    y_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/TF-IDF/y_test.csv")
    return X_train,X_test,y_train,y_test

import pandas as pd
import numpy as np

def get_Word2Vec_train_test_data():
    # Load the Word2Vec sentence embeddings (assuming they are already in CSV files)
    X_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/Word2Vec/x_train.csv")
    X_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/Word2Vec/x_test.csv")
    y_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/Word2Vec/y_train.csv")
    y_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/Word2Vec/y_test.csv")
    
    # Convert DataFrame to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # Convert y_train and y_test to numpy arrays
    y_train = y_train.to_numpy().flatten()  # Flatten to 1D array if necessary
    y_test = y_test.to_numpy().flatten()  # Flatten to 1D array if necessary
    
    # Reshape X_train and X_test for LSTM input (samples, timesteps, features)
    # Since we are using sentence embeddings of size 128, we reshape to (samples, 128, 1)
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test


def get_GloVe_train_test_data():
    # Load the Word2Vec sentence embeddings (assuming they are already in CSV files)
    X_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/GloVe/x_train.csv")
    X_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/GloVe/x_test.csv")
    y_train = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/GloVe/y_train.csv")
    y_test = pd.read_csv("./../../../2 Transform/Lyrics/2 Train Test Set/Output/GloVe/y_test.csv")
    
    # Convert DataFrame to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    
    # Convert y_train and y_test to numpy arrays
    y_train = y_train.to_numpy().flatten()  # Flatten to 1D array if necessary
    y_test = y_test.to_numpy().flatten()  # Flatten to 1D array if necessary
    
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test
