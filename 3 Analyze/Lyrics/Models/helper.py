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
