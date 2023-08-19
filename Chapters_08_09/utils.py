from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np


def plot_info(train_losses, val_accuracies):

    # plot the training loss
    plt.plot(train_losses, marker='o')
    # change the values on the ticks, starting from 1
    plt.xticks(np.arange(len(train_losses)), np.arange(1, len(train_losses) + 1))
    plt.grid()
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training Loss'])
    plt.show()


    # plot the validation accuracy
    plt.plot(val_accuracies, marker='o')
    # change the values on the ticks, starting from 1
    plt.xticks(np.arange(len(val_accuracies)), np.arange(1, len(val_accuracies) + 1))
    plt.axis(ymin=0.0, ymax=100)
    plt.grid()
    plt.title('Validation Accuracy (%)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Validation Accuracy'])
    plt.show()


def plot_confusion_matrix(ground_truth, predictions):
    # constant for classes
    classes = ('Intact', 'Accident')
    cf_matrix = confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(2, 2))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
