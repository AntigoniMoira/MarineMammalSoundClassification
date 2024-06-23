import os
import numpy as np
import librosa

def ensure_dir(directory):
    """
    Ensure that a directory exists. If it does not exist, create it.
    
    Parameters:
    directory (str): The path to the directory to ensure.
    
    Returns:
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_wav_duration(file_path):
    """
    Get the duration of a .wav audio file.
    
    Parameters:
    file_path (str): The path to the .wav audio file.
    
    Returns:
    float: The duration of the audio file in seconds, rounded to 2 decimal places.
    """
    y, sr = librosa.load(file_path)
    duration = librosa.get_duration(y=y, sr=sr)
    return round(duration, 2)

def calculate_acc_and_f1_from_cm(cm):
    """
    Calculate accuracy and macro-averaged F1 score from a confusion matrix.

    Parameters:
    cm (numpy.ndarray): Confusion matrix of shape (n_classes, n_classes).

    Returns:
    tuple: A tuple containing:
        - accuracy (float): The overall accuracy of the model.
        - f1_macro (float): The macro-averaged F1 score across all classes.
    """
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Calculate precision and recall for each class
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    
    # Calculate F1 score for each class
    f1_per_class = 2 * (precision * recall) / (precision + recall)
    f1_per_class = np.nan_to_num(f1_per_class)  # Replace NaN with 0
    
    # Calculate macro-averaged F1 score
    f1_macro = np.mean(f1_per_class)
    
    return accuracy, f1_macro
