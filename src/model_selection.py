from scipy.sparse import data
from utility import read_monk_dataset
import numpy as np
import tqdm

# TODO implement cross-validation, grid search and random search inside this class

# split a dataset into a train and validation set
def holdout_validation(net,dataset, labels, test_size, loss, metr, lr, shuffle=True, lr_decay=None, limit_step=None, decay_rate=None, decay_steps=None,
            momentum=0., nesterov=False, epochs=1, batch_size=1, strip=0, reg_type='l2', lambda_=0):
    train_size = int((1 - test_size) * len(dataset))
    if shuffle:
        # shuffle the whole dataset once
        indexes = list(range(len(dataset)))
        np.random.shuffle(indexes)
        dataset = dataset[indexes]
        labels = labels[indexes]
    train_X, val_X = dataset[:train_size,:], dataset[train_size:,:]
    train_Y, val_Y = labels[:train_size,:], labels[train_size:,:]
    net.compile(error_func=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                    decay_rate=decay_rate, decay_steps=decay_steps, momentum=momentum,
                    nesterov=nesterov, reg_type=reg_type, lambda_=lambda_)
    tr_err, tr_metr, val_err, val_metr = net.fit(tr_x=train_X, tr_y=train_Y, val_x=val_X, val_y=val_Y, batch_size=batch_size, epochs=epochs, strip_early_stopping=strip)
    return tr_err, tr_metr, val_err, val_metr

#TODO DocString documentation
def kfold_CV(net, dataset, loss, metr, lr, path=None, lr_decay=None, limit_step=None, decay_rate=None, decay_steps=None,
            momentum=0., nesterov=False, epochs=1, batch_size=1, k_folds=5, reg_type='l2', lambda_=0,
            disable_tqdms=(True, True), plot=True, verbose=False, **kwargs):
    # TODO implement utility function to read cup dataset
    if dataset == "cup":
        # dev_set_x, labels, _, _, _ = read_cup(int_ts=True)
        print("cup")
    else:
        rescale = True if net.params['act_functions'][-1] == 'tanh' else False
        dev_set_x, labels = read_monk_dataset(dataset=dataset, rescale=rescale)

    # split the dataset into folds
    x_folds = np.array(np.array_split(dev_set_x, k_folds), dtype=object)
    y_folds = np.array(np.array_split(labels, k_folds), dtype=object)

    # initialize vectors for plots
    tr_error_values, tr_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_error_values, val_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_metric_per_fold, val_error_per_fold = [], []
    tr_error_per_fold, tr_metric_per_fold = [], []

    # CV cycle
    for i in tqdm.tqdm(range(k_folds), desc='Iterating over folds', disable=disable_tqdms[0]):
        # create validation set and training set using the folds (for one iteration of CV)
        tr_data, tr_targets, val_data, val_targets = sets_from_folds(x_folds, y_folds, val_fold_index=i)

        # compile and fit the model on the current training set and evaluate it on the current validation set
        net.compile(opt=opt, error_func=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                    decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase, momentum=momentum,
                    reg_type=reg_type, lambd=lambd)
        warnings.simplefilter("error")
        try:
            tr_history = net.fit(tr_x=tr_data, tr_y=tr_targets, val_x=val_data, val_y=val_targets, epochs=epochs,
                                 batch_size=batch_size, disable_tqdm=disable_tqdms[1])
        except Exception as e:
            print(f"{e.__class__.__name__} occurred. Training suppressed.")
            print(e, '\n')
            return

        # metrics for the graph
        # composition of tr_history:
        #   [0] --> training error values for each epoch
        #   [1] --> training metric values for each epoch
        #   [2] --> validation error values for each epoch
        #   [3] --> validation metric values for each epoch
        tr_error_values += tr_history[0]
        tr_metric_values += tr_history[1]
        val_error_values += tr_history[2]
        val_metric_values += tr_history[3]
        try:
            tr_error_per_fold.append(tr_history[0][-1])
            tr_metric_per_fold.append(tr_history[1][-1])
            val_error_per_fold.append(tr_history[2][-1])
            val_metric_per_fold.append(tr_history[3][-1])
        except TypeError:
            tr_error_per_fold.append(tr_history[0])
            tr_metric_per_fold.append(tr_history[1])
            val_error_per_fold.append(tr_history[2])
            val_metric_per_fold.append(tr_history[3])

        # reset net's weights for the next iteration of CV
        net = Network(**net.params)

    # average the validation results of every fold
    tr_error_values /= k_folds
    tr_metric_values /= k_folds
    val_error_values /= k_folds
    val_metric_values /= k_folds

    # results
    avg_val_err, std_val_err = np.mean(val_error_per_fold), np.std(val_error_per_fold)
    avg_val_metric, std_val_metric = np.mean(val_metric_per_fold), np.std(val_metric_per_fold)
    avg_tr_err, std_tr_err = np.mean(tr_error_per_fold), np.std(tr_error_per_fold)
    avg_tr_metr, std_tr_metr = np.mean(tr_metric_per_fold), np.std(tr_metric_per_fold)

    # print k-fold metrics
    if verbose:
        print("\nScores per fold:")
        for i in range(k_folds):
            print(f"Fold {i + 1}:\nVal Loss: {val_error_per_fold[i]} - Val Metric: {val_metric_per_fold[i]}\n"
                  f"Train Loss: {tr_error_per_fold[i]} - Train Metric: {tr_metric_per_fold[i]}\n{'-' * 62}\n")
        print('\nAverage scores for all folds:')
        print("Val Loss: {} - std:(+/- {})\n"
              "Train Loss: {} - std:(+/- {})\n"
              "Val Metric: {} - std:(+/- {})\n"
              "Train Metric: {} - std(+/- {}\n".format(avg_val_err, std_val_err,
                                                       avg_tr_err, std_tr_err,
                                                       avg_val_metric, std_val_metric,
                                                       avg_tr_metr, std_tr_metr))
    if plot:
        ylim, lbltr, lblval = None, None, None
        if "monk" in dataset:
            ylim, lbltr, lblval = (0., 1.1), "Training", "Validation"
        plot_curves(tr_error_values, val_error_values, tr_metric_values, val_metric_values, path, ylim=ylim,
                    lbltr=lbltr, lblval=lblval)
    return avg_val_err, std_val_err, avg_val_metric, std_val_metric
    
