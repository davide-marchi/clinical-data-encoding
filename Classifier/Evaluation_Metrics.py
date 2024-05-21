from sklearn import metrics
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scikitplot as skplt

from matplotlib import pyplot as plt


def plot_c_matrix(test_label, test_pred, classifier_name):
    """
    Plots a confusion matrix for the given test labels and predictions.
    
    Args:
        test_label (array-like): True labels of the test data.
        test_pred (array-like): Predicted labels of the test data.
        classifier_name (str): Name of the classifier (used for the plot title).
    
    Returns:
        None
    """
    # Compute the confusion matrix and create a ConfusionMatrixDisplay object
    cm = confusion_matrix(test_label, test_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    disp.plot()
    plt.title('Confusion matrix of ' + classifier_name)
    # Display the plot
    plt.show()


def report_scores(test_label, test_pred, labels=None):
    """
    Prints the classification report for the given test labels and predictions.
    
    Args:
        test_label (array-like): True labels of the test data.
        test_pred (array-like): Predicted labels of the test data.
        labels (array-like, optional): List of label indices to include in the report.
    
    Returns:
        None
    """
    # Print the classification report using sklearn's classification_report function
    print(classification_report(test_label, test_pred, labels=labels))
    
def plot_loss(train_losses, val_losses=None, figsize=(8,6), print_every=1):
    """
    Plot training and validation losses.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses (optional).
        figsize (tuple): Size of the figure.
        print_every (int): Interval for printing training progress.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.xticks(epochs[::print_every])
    plt.show()
    
def plot_accuracy(train_accuracies, val_accuracies=None, figsize=(8,6), print_every=1):
    """
    Plot training and validation accuracies.

    Args:
        train_accuracies (list): List of training accuracies.
        figsize (tuple): Size of the figure.
        val_accuracies (list): List of validation accuracies (optional).
        print_every (int): Interval for printing training progress.
    """
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    if val_accuracies:
        plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.xticks(epochs[::print_every])
    plt.show()