import os
import librosa
import numpy as np
import pickle
import scipy
from .utilities import ensure_dir

def segment_audio(file_path, segment_length=2.0):
    """
    Segments an audio file into equal parts of specified length with zero padding for the last segment if needed.

    Parameters:
    file_path (str): Path to the input audio file.
    segment_length (float): Length of each segment in seconds. Default is 2.0 seconds.

    Returns:
    list: List of audio segments.
    int: Sample rate of the audio file.
    """
    y, sr = librosa.load(file_path, sr=None)
    segment_samples = int(segment_length * sr)
    total_samples = len(y)

    segments = []
    for start in range(0, total_samples, segment_samples):
        end = start + segment_samples
        segment = y[start:end]
        if len(segment) < segment_samples:
            # Zero-pad the last segment if needed
            segment = np.pad(segment, (0, segment_samples - len(segment)), 'constant')
        segments.append(segment)

    return segments, sr


def create_spectrogram(segment):
    """
    Creates a spectrogram for a given audio segment.

    Parameters:
    segment (ndarray): Audio segment data.

    Returns:
    ndarray: Spectrogram data in decibels.
    """
    S = librosa.stft(segment)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    return S_db

def create_melgram(segment, sr):
    """
    Creates a mel-spectrogram for a given audio segment.

    Parameters:
    segment (ndarray): Audio segment data.
    sr (int): Sample rate of the audio segment.

    Returns:
    ndarray: Mel-spectrogram data in decibels.
    """
    S = librosa.feature.melspectrogram(y=segment, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def resize_spectrogram(data, n):
    """
    Resize a spectrogram to a desired n x n size.

    Parameters:
    data (np.array): Input spectrogram with shape (height, width).
    n (int): Desired size for the output spectrogram (n x n).

    Returns:
    np.array: Resized spectrogram with shape (n, n).
    """
    # Calculate the current size of the spectrogram
    current_height, current_width = data.shape

    # Calculate the zoom factors for height and width to reach the desired size
    zoom_factor_height = n / current_height
    zoom_factor_width = n / current_width

    # Resize the spectrogram
    D_resized = scipy.ndimage.zoom(data, (zoom_factor_height, zoom_factor_width))

    return D_resized

def create_pkl_with_spectrograms(set_name, seg_dur, n_size=128):
    """
    Create pickle files containing spectrograms, mel-spectrograms, class labels, and segment counts
    from audio files segmented into equal parts with zero padding if needed.

    Parameters:
    set_name (str): Name of the dataset directory (train/val/test) located in the 'data_split' directory.
    seg_dur (float): Duration of each segment in seconds.
    n_size (int): Desired size for the output spectrograms and mel-spectrograms (n x n).

    The function generates and saves the following pickle files:
    - 'spectrograms/{seg_dur}_secs/{set_name}_specs.pkl': List of resized spectrograms.
    - 'spectrograms/{seg_dur}_secs/{set_name}_mels.pkl': List of resized mel-spectrograms.
    - 'spectrograms/{seg_dur}_secs/{set_name}_classes.pkl': List of class labels corresponding to the segments.
    - 'spectrograms/{seg_dur}_secs/{set_name}_segments.pkl': List of segment counts per file.
    """
    set_dir = os.path.join("data_split", set_name)
    set_classes = os.listdir(set_dir)
    dirs = [os.path.join(set_dir, c) for c in set_classes]

    spectrograms = []
    melgrams = []
    classes = []
    seg_per_file = []

    # Process each class directory
    for d in dirs:
        class_name = os.path.basename(d)  # Class name (directory name)
        files_list = os.listdir(d)  # List of audio files in the class directory
        for file_name in files_list:
            file_path = os.path.join(d, file_name)  # Full path to the audio file
            segments, sr = segment_audio(file_path, seg_dur)  # Segment the audio file
            seg_per_file.append(len(segments)) # Calculate the number of segments for the audio file

            # Process each segment
            for i, segment in enumerate(segments):
                spectrogram = create_spectrogram(segment)
                # r_spectrogram = resize_spectrogram(spectrogram, n_size)
                melgram = create_melgram(segment, sr)
                # r_melgram = resize_spectrogram(melgram, n_size)
                classes.append(class_name)
                spectrograms.append(spectrogram)
                melgrams.append(melgram)

    sub_folder = os.path.join('spectrograms', str(seg_dur) + '_secs')
    ensure_dir(sub_folder)

    # Save spectrograms to a pickle file
    with open(os.path.join(sub_folder, set_name + '_specs.pkl'), 'wb') as f:
        pickle.dump(spectrograms, f)

    # Save mel-spectrograms to a pickle file
    with open(os.path.join(sub_folder, set_name + '_mels.pkl'), 'wb') as f:
        pickle.dump(melgrams, f)

    # Save class labels to a pickle file
    with open(os.path.join(sub_folder, set_name + '_classes.pkl'), 'wb') as f:
        pickle.dump(classes, f)

    # Save segment counts per file to a pickle file
    with open(os.path.join(sub_folder, set_name + '_segments.pkl'), 'wb') as f:
        pickle.dump(seg_per_file, f)
