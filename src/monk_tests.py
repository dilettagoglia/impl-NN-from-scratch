from model_selection import *
from net import Network
from utility import read_monk_dataset
import numpy as np

if __name__ == '__main__':
    # create directory for plots
    dir_name = "../plots/"

    # training and test set - {monk1 - monk2 - monk3}
    ds_monk2 = "monks-2.train"
    ds_monk1 = "monks-1.train"
    ds_monk3 = "monks-3.train"

    ts_monk1 = "monks-1.test"
    ts_monk2 = "monks-2.test"
    ts_monk3 = "monks-3.test"

    # read monk 1 {train - test}
    monk_train1, labels1 = read_monk_dataset(ds_monk1, rescale=False)
    monk_test1, labels_ts1 = read_monk_dataset(ts_monk1, rescale=False)

    # read monk 2 {train - test}
    monk_train2, labels2 = read_monk_dataset(ds_monk2, rescale=False)
    monk_test2, labels_ts2 = read_monk_dataset(ts_monk2, rescale=False)

    # monk 3 {train - test}
    monk_train3, labels3 = read_monk_dataset(ds_monk3, rescale=False)
    monk_test3, labels_ts3 = read_monk_dataset(ts_monk3, rescale=False)

    # model parameters
    mod_par_monk1 = {
        'input_dim': 17,
        'units_per_layer': (4, 1),
        'act_functions': ('relu', 'sigmoid'),
        'weights_init': 'random',
        'init_value': 0.2,
        #'limits': (-0.1, 0.1)
    }

    mod_par_monk2 = {
        'input_dim': 17,
        'units_per_layer': (4, 1),
        'act_functions': ('relu', 'sigmoid'),
        'weights_init': 'random',
        'init_value': 0.2,
        #'limits': (-0.25, 0.25)
    }

    mod_par_monk3 = {
        'input_dim': 17,
        'units_per_layer': (15, 1),
        'act_functions': ('relu', 'sigmoid'),
        'weights_init': 'random',
        # 'init_value': 0.2,
        #'limits': (-0.1, 0.1)
    }

    # create model
    model_monk1 = Network(**mod_par_monk1)
    model_monk2 = Network(**mod_par_monk2)
    model_monk3 = Network(**mod_par_monk3)

    # training parameters
    params_monk1 = {'lr': 0.76, 'momentum': 0.83, 'epochs': 500, 'batch_size': 'full',
                    'loss': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk2 = {'lr': 0.8, 'momentum': 0.8, 'epochs': 500, 'batch_size': 'full',
                    'loss': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk3_noreg = {'lr': 0.8, 'momentum': 0.8, 'epochs': 500, 'batch_size': 'full',
                          'loss': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk3_l2 = {'lr': 0.8, 'momentum': 0.8, 'epochs': 500, 'batch_size': 'full', 'lambd': 0.0023,
                       'reg_type': 'l2', 'loss': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk3_l1 = {'lr': 0.8, 'momentum': 0.8, 'epochs': 500, 'batch_size': 'full', 'lambd': 0.0012,
                       'reg_type': 'l1', 'loss': 'squared_error', 'metr': 'binary_class_accuracy'}

    # Average prediction results
    avg_tr_error, avg_tr_acc, avg_ts_error, avg_ts_acc = [], [], [], []

    # model selection monks
    '''
    kfold_CV(net=model_monk1, dataset='monks-1.train', **params_monk1, k_folds=5, verbose=True, plot=True,
            disable_tqdms=(True, False))
    '''

    grid_search('monks-1.train', **params_monk1, coarse=True, n_config=1)