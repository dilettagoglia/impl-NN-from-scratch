import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
# import scikitplot as skplt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split # hold-out approach
from tqdm import tqdm
# from IPython import display

"""
Read Dataset
"""

def read_monk_dataset(dataset):
    """
    Reads the monks datasets and performs a preliminary preproccessing of data.
    Creates the labels for supervised classification and hide them to the classifier.

    Attr:
        name: name of the dataset

    Return:
        monk dataset and labels (as numpy ndarrays)
    """

    col_names = ['class', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'Id']

    # Read the .csv file containing the data. The first line contains the list of attributes. The data is assigned to a Pandas dataframe.
    monk_train_1 = pd.read_csv(f"../datasets/monks/{str(dataset)}", sep=" ", names=col_names)
    monk_train_1.set_index('Id', inplace=True)

    # Labels creation - Dropping the "class" column from the Monk dataset: this represents the target y.
    labels = monk_train_1['class']
    monk_train_1.drop(columns=['class'], inplace=True)
    labels = pd.Series(labels).to_numpy()  # from pd Series into numpy array
    labels = np.expand_dims(labels, 1)  # add a flat dimension

    """One-Hot Encoding"""
    encoder = OneHotEncoder().fit(monk_train_1)
    monk_dataset_1 = encoder.transform(monk_train_1).toarray()

    return monk_dataset_1, labels

""" 
Visualization 
"""

def plot_curves(tr_loss, val_loss, tr_metr, val_metr, path=None, ylim=(0., 1.1), lbltr='development',
                lblval='internal test', *args):
    """
    Plot the curves of training loss, training metric, validation loss, validation metric
    :param tr_loss: vector with the training error values
    :param val_loss: vector with the validation error values
    :param tr_metr: vector with the training metric values
    :param val_metr: vector with the validation metric values
    :param path: if not None, path where to save the plot (otherwise it will be displayed)
    :param ylim: value for "set_ylim" of pyplot
    :param lbltr: label for the training curve
    :param lblval: label for the validation curve
    """
    figure, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(len(tr_loss)), tr_loss, color='b', linestyle='dashed', label=lbltr)
    ax[0].plot(range(len(val_loss)), val_loss, color='r', label=lblval)
    ax[0].legend(loc='best', prop={'size': 9})
    ax[0].set_xlabel('Epochs', fontweight='bold')
    ax[0].set_ylabel('Error', fontweight='bold')
    ax[0].grid()
    ax[1].plot(range(len(tr_metr)), tr_metr, color='b', linestyle='dashed', label=lbltr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lblval)
    ax[1].legend(loc='best', prop={'size': 9})
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Acc', fontweight='bold')
    ax[1].set_ylim(ylim)
    ax[1].grid()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)

