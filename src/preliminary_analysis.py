import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from utility import *

# Monk train dataset analysis
ds_names=['monks-1.train', 'monks-2.train', 'monks-3.train']
for dataset in ds_names:
    print('\n\n **** Analysis of "',dataset,'" dataset ****')
    monk_train, labels_tr = read_monk_dataset(dataset, preliminary_analysis=True)
    print('One-Hot encoded dataset. \n Training set:\n', monk_train, '\n Labels:\n', labels_tr.T)
    # Check dimensions
    print('\nLabels vector shape:\n', labels_tr.shape, '\n Training dataset shape: (examples, features)\n', monk_train.shape)


# Cup train dataset analysis

print('\n\n **** Analysis of ML-CUP20-TR dataset ****')
tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data = read_cup(int_ts=True)
# Check dimensions
print('\nLabels vector shape:\n', tr_targets.shape, '\n Training dataset shape: (examples, features)\n', tr_data.shape)
