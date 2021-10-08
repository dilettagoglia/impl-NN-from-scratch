import statistics
import math
import os
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

#TODO use DocString documentation for this file
"""
Read Dataset
"""


def read_monk_dataset(dataset, rescale=False, preliminary_analysis=None):

    """
    Reads the monks datasets and performs a preliminary preproccessing of data.
    Creates the labels for supervised classification and hide them to the classifier.

    Attr:
        dataset: name of the dataset
        rescale: rescale to [-1,+1] the targets

    Return:
        monk dataset and labels (as numpy ndarrays)
    """


    # Read the .csv file containing the data. The first line contains the list of attributes. The data is assigned to a Pandas dataframe.
    col_names = ['class', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'Id']
    monk_train = pd.read_csv(f"../datasets/monks/{str(dataset)}", sep=" ", names=col_names)
    monk_train.set_index('Id', inplace=True)

    if preliminary_analysis:

        # Class group plot
        monk_train.groupby(monk_train['class']).sum().plot(kind="bar")
        plt.title(f"Dataset: {str(dataset)}")
        plt.xlabel("Class")
        plt.ylabel("Number of data points")
        plt.show()

        # Check class distribution
        print("\n Check if dataset is balanced (class distribution):\n",monk_train.groupby(monk_train['class']).sum(),"\n",)

    # Labels creation - Dropping the "class" column from the Monk dataset: this represents the target y.

    labels = monk_train['class']
    if rescale:
        labels[labels == 0] = -1 # rescale to -1 for TanH output function

    monk_train.drop(columns=['class'], inplace=True)

    labels = pd.Series(labels).to_numpy()  # from pd Series into numpy array
    labels = np.expand_dims(labels, 1)  # add a flat dimension

    """One-Hot Encoding"""
    encoder = OneHotEncoder().fit(monk_train)
    monk_dataset = encoder.transform(monk_train).toarray()

    # choose class values: [-1, 1] or [0, 1]
    if rescale:
        labels[labels == 0] = -1

    # shuffle the whole dataset once
    indexes = list(range(len(monk_dataset)))
    np.random.shuffle(indexes)
    monk_dataset = monk_dataset[indexes]
    labels = labels[indexes]

    return monk_dataset, labels

def read_cup(int_ts=False):
    """
    Reads the CUP training and test set
    :return: CUP training data, CUP training targets and CUP test data (as numpy ndarray)
    """
    # read the dataset
    col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
    directory = "../datasets/ML-cup/"
    int_ts_path = directory + "CUP-INTERNAL-TEST.csv"
    dev_set_path = directory + "CUP-DEV-SET.csv"
    file = "ML-CUP20-TR.csv"

    if int_ts and not (os.path.exists(int_ts_path) and os.path.exists(dev_set_path)):
        df = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(0, 13))
        df = df.sample(frac=1, axis=0) # shuffle the dataset
        int_ts_df = df.iloc[: math.floor(len(df) * 0.2), :]
        dev_set_df = df.iloc[math.floor(len(df) * 0.2) :, :]
        int_ts_df.to_csv(path_or_buf=int_ts_path, index=False, float_format='%.6f', header=False)
        dev_set_df.to_csv(path_or_buf=dev_set_path, index=False, float_format='%.6f', header=False)

    if int_ts and os.path.exists(int_ts_path) and os.path.exists(dev_set_path):
        tr_data = pd.read_csv(dev_set_path, sep=',', names=col_names, usecols=range(1, 11))
        tr_targets = pd.read_csv(dev_set_path, sep=',', names=col_names, usecols=range(11, 13))
        int_ts_data = pd.read_csv(int_ts_path,  sep=',', names=col_names, usecols=range(1, 11))
        int_ts_targets = pd.read_csv(int_ts_path,  sep=',', names=col_names, usecols=range(11, 13))
        int_ts_data = int_ts_data.to_numpy(dtype=np.float32)
        int_ts_targets = int_ts_targets.to_numpy(dtype=np.float32)
    else:
        tr_data = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
        tr_targets = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))

    file = "ML-CUP20-TS.csv"
    cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[: -2], skiprows=range(7), usecols=range(1, 11))

    tr_data = tr_data.to_numpy(dtype=np.float32)
    tr_targets = tr_targets.to_numpy(dtype=np.float32)
    cup_ts_data = cup_ts_data.to_numpy(dtype=np.float32)

    # shuffle the training dataset once
    # (pd.read_csv in previous instructions sorts by ID)
    indexes = list(range(tr_targets.shape[0]))
    np.random.shuffle(indexes)
    tr_data = tr_data[indexes]
    tr_targets = tr_targets[indexes]

    # detach internal test set if needed (maybe some problem with OS file reading)
    if int_ts:
        if not (os.path.exists(int_ts_path) and os.path.exists(dev_set_path)):
            tr_data, int_ts_data, tr_targets, int_ts_targets = train_test_split(tr_data, tr_targets, test_size=0.2)

        return tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data

    return tr_data, tr_targets, cup_ts_data

def sets_from_folds(x_folds, y_folds, val_fold_index):
    """ Takes folds from cross validation and return training and validation sets as a whole (not lists of folds)

    Args:
        x_folds (np.ndarray): list of folds containing the data
        y_folds (np.ndarray): list of folds containing the targets
        val_fold_index (int): index of the fold to use as validation set

    Returns:
        [tuple of np.ndarray]: training data set, training targets set, validation data set, validation targets set
    """    
    val_data, val_targets = x_folds[val_fold_index], y_folds[val_fold_index]
    tr_data_folds = np.concatenate((x_folds[: val_fold_index], x_folds[val_fold_index + 1:]))
    tr_targets_folds = np.concatenate((y_folds[: val_fold_index], y_folds[val_fold_index + 1:]))
    # here tr_data_folds & tr_targets_folds are still a "list of folds", we need a single seq as a whole
    tr_data = tr_data_folds[0]
    tr_targets = tr_targets_folds[0]
    for j in range(1, len(tr_data_folds)):
        tr_data = np.concatenate((tr_data, tr_data_folds[j]))
        tr_targets = np.concatenate((tr_targets, tr_targets_folds[j]))
    tr_data = np.array(tr_data, dtype=np.float32)
    tr_targets = np.array(tr_targets, dtype=np.float32)
    val_data = np.array(val_data, dtype=np.float32) #TODO check if we can remove this instruction
    val_targets = np.array(val_targets, dtype=np.float32)
    return tr_data, tr_targets, val_data, val_targets

""" 
Visualization 
"""

def plot_curves(tr_loss, val_loss, tr_metr, val_metr, path=None, ylim=(0., 10.), lbltr='development',
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
    ax[0].set_ylim(ylim)
    ax[0].grid()
    ax[1].plot(range(len(tr_metr)), tr_metr, color='b', linestyle='dashed', label=lbltr)
    ax[1].plot(range(len(val_metr)), val_metr, color='r', label=lblval)
    ax[1].legend(loc='best', prop={'size': 9})
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Accuracy', fontweight='bold')
    ax[1].set_ylim(ylim)
    ax[1].grid()
    plt.suptitle(f'Error (TR/VL): {tr_loss[-1]} / {val_loss[-1]} | Accuracy% (TR/VL): {tr_metr[-1]} / {val_metr[-1]}')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

# replace 0 values with an extremely small value
def eta(x):
  ETA = 0.0000000001
  return np.maximum(x, ETA)

