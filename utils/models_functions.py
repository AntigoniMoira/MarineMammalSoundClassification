import pickle
import statistics
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def load_data(data_path, classes_path, normalization=False):
    """
    Loads and preprocesses spectrogram data and class labels from pickle files.

    Parameters:
    data_path (str): Path to the pickle file containing spectrogram data.
    classes_path (str): Path to the pickle file containing class labels.

    Returns:
    tuple: A tuple containing:
        - X (np.array): Normalized spectrogram data.
        - y (np.array): One-hot encoded class labels.
        - encoder (LabelEncoder): Fitted LabelEncoder instance for decoding labels.
    """
    with open(data_path, 'rb') as f:
        X = pickle.load(f)
    with open(classes_path, 'rb') as f:
        classes = pickle.load(f)

    if normalization:
        # Normalize spectrograms
        X = X / np.min(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(classes)
    y = to_categorical(y, num_classes=28)

    return np.array(X), np.array(y), encoder

def train_model(model, trainX, trainY, valX, valY, epochs=20, batch_size=32, patience=5):
    """
    Trains the given model using the provided training and validation data.

    Args:
    model (tensorflow.keras.Model): The Keras model to be trained.
    trainX (numpy.ndarray): Training data features.
    trainY (numpy.ndarray): Training data labels.
    valX (numpy.ndarray): Validation data features.
    valY (numpy.ndarray): Validation data labels.
    epochs (int, optional): The number of epochs to train the model. Defaults to 20.
    batch_size (int, optional): The batch size to use during training. Defaults to 32.
    patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.

    Returns:
    tensorflow.keras.callbacks.History: The history object that holds training and validation loss and accuracy values.
    """
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_lr=1e-5)

    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr])

    return history

def majority_voting(test_predictions, test_true, seg_per_file):
    """
    Reunifies segment-level predictions and true labels into file-level predictions and true labels
    by taking the most common value (mode) of the segments corresponding to each file.

    Parameters:
    test_predictions (list): List of predicted labels for each segment.
    test_true (list): List of true labels for each segment.
    seg_per_file (list): List of segment counts per file, indicating how many segments belong to each file.

    Returns:
    file_predictions (list): List of the most common predicted labels for each file.
    file_true (list): List of the most common true labels for each file.
    """
    index = 0
    file_predictions = []
    file_true = []

    for i in range(len(seg_per_file)):
        # Calculate the new index by adding the number of segments for the current file
        new_index = index + seg_per_file[i]

        # Determine the most common prediction and true label for the current file segments
        most_common_prediction = statistics.mode(test_predictions[index:new_index])
        most_common_true = statistics.mode(test_true[index:new_index])

        # Append the most common values to the file-level lists
        file_predictions.append(most_common_prediction)
        file_true.append(most_common_true)

        # Update the index to the start of the next file's segments
        index = new_index

    return file_predictions, file_true

def evaluate_model(model, testX, testY, testSegments):
    """
    Evaluates the given model using the provided test data, and computes metrics
    both at the segment level and at the file level by reunifying segments.

    Args:
    model (tensorflow.keras.Model): The Keras model to be evaluated.
    testX (numpy.ndarray): Test data features.
    testY (numpy.ndarray): Test data labels (one-hot encoded).
    testSegments (list): List indicating the number of segments per file.

    Returns:
    tuple: A tuple containing:
        - conf_matrix (numpy.ndarray): The confusion matrix of the segment-level test predictions.
        - accuracy (float): The accuracy score of the segment-level test predictions.
        - f1 (float): The F1 score of the segment-level test predictions.
        - reunify_conf_matrix (numpy.ndarray): The confusion matrix of the file-level test predictions.
        - reunify_accuracy (float): The accuracy score of the file-level test predictions.
        - reunify_f1 (float): The F1 score of the file-level test predictions.
    """
    # Make predictions on the test data
    test_predictions = np.argmax(model.predict(testX), axis=1)
    test_true = np.argmax(testY, axis=1)

    # Calculate segment-level metrics
    conf_matrix = confusion_matrix(test_true, test_predictions)
    accuracy = accuracy_score(test_true, test_predictions)
    f1 = f1_score(test_true, test_predictions, average='macro')

    # Majority voting to get file-level predictions
    test_predictions, test_true = majority_voting(test_predictions, test_true, testSegments)

    # Calculate file-level metrics
    reunify_conf_matrix = confusion_matrix(test_true, test_predictions)
    reunify_accuracy = accuracy_score(test_true, test_predictions)
    reunify_f1 = f1_score(test_true, test_predictions, average='macro')

    return conf_matrix, accuracy, f1, reunify_conf_matrix, reunify_accuracy, reunify_f1

def save_model(model, file_path):
    """
    Saves the given model to the specified file path.

    Args:
    model (tensorflow.keras.Model): The Keras model to be saved.
    file_path (str): The path where the model will be saved.
    """
    model.save(file_path)

def load_model(file_path):
    """
    Loads a Keras model from the specified file path.

    Args:
    file_path (str): The path from where the model will be loaded.

    Returns:
    tensorflow.keras.Model: The loaded Keras model.
    """
    return tf.keras.models.load_model(file_path)

def create_CNN_model(input_shape, num_classes):
    """
    Creates and compiles a Convolutional Neural Network (CNN) model.

    Parameters:
    input_shape (tuple): Shape of the input data (height, width, channels).
    num_classes (int): Number of output classes.

    Returns:
    tensorflow.keras.Model: The compiled CNN model.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape)),
    model.add(layers.MaxPooling2D(pool_size=(2, 2))),
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu')),
    model.add(layers.MaxPooling2D(pool_size=(2, 2))),
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Flatten()),
    model.add(layers.Dense(128, activation='relu')),
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_LSTM_model(input_shape, num_classes):
    """
    Creates and compiles a Long Short-Term Memory (LSTM) model.

    Parameters:
    input_shape (tuple): Shape of the input data (timesteps, features).
    num_classes (int): Number of output classes.

    Returns:
    tensorflow.keras.Model: The compiled LSTM model.
    """
    
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.1))
    
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.1))
    
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.1))
    
    # model.add(layers.Flatten())
    # model.add(tf.keras.layers.Reshape((-1, 128)))  # Reshape for LSTM layers
    
    # model.add(layers.LSTM(128, return_sequences=True))
    
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(128))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model