import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import scikitplot as skplt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm # interessante, leggere più documentazione
from IPython import display

from utility import *

# Monk train dataset analysis
ds_names=['monks-1.train', 'monks-2.train', 'monks-3.train']
for dataset in ds_names:
    print('\n\n **** Analysis of "',dataset,'" dataset ****')
    monk_train, labels_tr = read_monk_dataset(dataset, preliminary_analysis=True)
    print('One-Hot encoded dataset. \n Training set:\n', monk_train, '\n Labels:\n', labels_tr.T)
    # Check dimensions
    print('\nLabels vector shape:\n', labels_tr.shape, '\n Training dataset shape: (examples, features)\n', monk_train.shape)


