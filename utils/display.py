import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

def display_spectogram(data):
    """
    Displays a spectrogram from given spectrogram data.

    Parameters:
    data (ndarray): Spectrogram data to display.
    """
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(data, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()

def plot_history(hist):
    """
    Plots the training and validation accuracy from a Keras model's training history.

    Parameters:
    hist (tensorflow.keras.callbacks.History): History object returned by the `fit` method of a Keras model.

    The function generates a plot showing the training and validation accuracy over epochs.
    """
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, '-', label='Training Accuracy')
    plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()

def display_model_evaluation(model, subtitle, history, accuracy, f1_score, conf_matrix, labels, majority_vote=False,  majority_conf_matrix=[], majority_accuracy=0, majority_f1_score=0):
    """
    Display a summary of a deep learning model's evaluation results, including training history, confusion matrix, 
    accuracy, and F1 score. Optionally includes results for majority voting ensemble.

    Parameters:
    model : keras.Model
        The trained deep learning model.
    subtitle : str
        Subtitle to be displayed below the main title.
    history : dict
        Dictionary containing training history with keys 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
    accuracy : float
        Accuracy of the model on the evaluation dataset.
    f1_score : float
        F1 score of the model on the evaluation dataset.
    conf_matrix : ndarray
        Confusion matrix of the model's predictions.
    labels : list
        List of labels for the confusion matrix axes.
    majority_vote : bool, optional
        Whether to include a confusion matrix and metrics for majority voting ensemble (default is False).
    majority_conf_matrix : ndarray, optional
        Confusion matrix for majority voting ensemble predictions (default is an empty list).
    majority_accuracy : float, optional
        Accuracy for majority voting ensemble (default is 0).
    majority_f1_score : float, optional
        F1 score for majority voting ensemble (default is 0).

    Returns:
    None
    """
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(24, 14))

    # Plot confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', linewidth=.5, linecolor='blue', cmap='Blues', ax=axes[0, 0], xticklabels=labels, yticklabels=labels)
    axes[0, 0].set_title('Confusion Matrix', y=1.10)
    axes[0, 0].set_xlabel('Predicted Label')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].text(5, -0.8, f'Accuracy: {accuracy*100:.1f}%', fontsize=12)
    axes[0, 0].text(15, -0.8, f'F1 Score: {f1_score*100:.1f}%', fontsize=12)

    if majority_vote:
        # Plot confusion matrix majority
        sns.heatmap(majority_conf_matrix, annot=True, fmt='d', linewidth=.5, linecolor='blue', cmap='Blues', ax=axes[0, 1], xticklabels=labels, yticklabels=labels)
        axes[0, 1].set_title('Majority Voting Confusion Matrix', y=1.10)
        axes[0, 1].set_xlabel('Predicted Label')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].text(5, -0.8, f'Accuracy: {majority_accuracy*100:.1f}%', fontsize=12)
        axes[0, 1].text(15, -0.8, f'F1 Score: {majority_f1_score*100:.1f}%', fontsize=12)

    # Plot model loss
    axes[1, 0].plot(history['loss'], '-', color="blue", label='Training Loss')
    axes[1, 0].plot(history['val_loss'], ':', color="blue", label='Validation Loss')
    axes[1, 0].set_title('Model Loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()

    # Plot model accuracy
    axes[1, 1].plot(history['accuracy'], '-', color="blue", label='Training Accuracy')
    axes[1, 1].plot(history['val_accuracy'], ':', color="blue", label='Validation Accuracy')
    axes[1, 1].set_title('Model Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()

    # Adding the main title and subtitle
    plt.suptitle(f'{model} Performance Summary', fontsize=16)
    fig.text(0.5, 0.95, subtitle, ha='center', fontsize=12, alpha=0.75)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle

    plt.show()
