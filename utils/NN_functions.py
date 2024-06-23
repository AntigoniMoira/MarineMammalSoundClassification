import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Functions for each step of the model training and evaluation process

def load_data(file_path, sep='\t'):
    """
    Loads data from a CSV file and prepares it for training the model.

    Args:
    file_path (str): The path to the CSV file containing the data.
    sep (str, optional): The delimiter of the CSV file. Defaults to '\t'.

    Returns:
    tuple: A tuple containing:
        - X (numpy.ndarray): The feature matrix.
        - y (numpy.ndarray): The one-hot encoded labels.
        - encoder (LabelEncoder): The label encoder fitted on the class labels.
    """
    df = pd.read_csv(file_path, sep=sep)
    X = np.array(df.iloc[:, 3:].values.tolist())
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['class'])
    y = to_categorical(y, num_classes=28)
    return X, y, encoder

def create_model(initial_dimensionality, input_shape, num_classes, batch_norm=True, dropout=False):
    """
    Creates a neural network model with decreasing dimensionality.

    Args:
    initial_dimensionality (int): The number of units in the first dense layer.
    input_shape (int): The shape of the input data.
    num_classes (int): The number of output classes.
    batch_norm (bool, optional): Whether to include batch normalization layers. Defaults to True.
    dropout (bool, optional): Whether to include dropout layers. Defaults to False.

    Returns:
    tensorflow.keras.Sequential: The compiled Keras model.
    """
    model = Sequential()
    model.add(layers.Dense(initial_dimensionality, activation='relu', input_shape=(input_shape,)))
    if batch_norm:
        model.add(layers.BatchNormalization())
    if dropout:
        model.add(layers.Dropout(0.1))

    dim = initial_dimensionality
    while dim > 64:
      dim //= 2
      model.add(layers.Dense(dim, activation='relu'))
      if batch_norm:
          model.add(layers.BatchNormalization())
      if dropout:
          model.add(layers.Dropout(0.1))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, trainX, trainY, valX, valY, epochs=100, batch_size=32, patience=5):
    """
    Trains the given model using the provided training and validation data.

    Args:
    model (tensorflow.keras.Model): The Keras model to be trained.
    trainX (numpy.ndarray): Training data features.
    trainY (numpy.ndarray): Training data labels.
    valX (numpy.ndarray): Validation data features.
    valY (numpy.ndarray): Validation data labels.
    epochs (int, optional): The number of epochs to train the model. Defaults to 100.
    batch_size (int, optional): The batch size to use during training. Defaults to 32.
    patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 5.

    Returns:
    tensorflow.keras.callbacks.History: The history object that holds training and validation loss and accuracy values.
    """
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, min_lr=1e-5)

    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stopping, reduce_lr])
    return history

def evaluate_model(model, testX, testY):
    """
    Evaluates the given model using the provided test data.

    Args:
    model (tensorflow.keras.Model): The Keras model to be evaluated.
    testX (numpy.ndarray): Test data features.
    testY (numpy.ndarray): Test data labels.

    Returns:
    tuple: A tuple containing:
        - conf_matrix (numpy.ndarray): The confusion matrix of the test predictions.
        - accuracy (float): The accuracy score of the test predictions.
        - f1 (float): The F1 score of the test predictions.
    """
    test_predictions = np.argmax(model.predict(testX), axis=1)
    test_true = np.argmax(testY, axis=1)
    conf_matrix = confusion_matrix(test_true, test_predictions)
    accuracy = accuracy_score(test_true, test_predictions)
    f1 = f1_score(test_true, test_predictions, average='macro')
    return conf_matrix, accuracy, f1


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


