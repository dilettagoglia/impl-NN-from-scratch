from model_selection import *
from net import Network
from utility import read_monk_dataset
import numpy as np

if __name__ == '__main__':
    # create directory for plots
    dir_name = "../plots/"

    # training and test set - {monk1 - monk2 - monk3}
    ds_monk1 = "monks-1.train"
    ds_monk2 = "monks-2.train"
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
        'units_per_layer': [4, 1],
        'act_functions': ['relu', 'sigmoid'],
        'weights_init': 'glorot'
    }

    mod_par_monk2 = {
        'input_dim': 17,
        'units_per_layer': [4, 1],
        'act_functions': ['relu', 'sigmoid'],
        'weights_init': 'glorot'
    }

    mod_par_monk3 = {
        'input_dim': 17,
        'units_per_layer': [4, 1],
        'act_functions': ['relu', 'sigmoid'],
        'weights_init': 'glorot'
    }

    # create model
    # model_monk1 = Network(**mod_par_monk1)
    # model_monk2 = Network(**mod_par_monk2)
    # model_monk3 = Network(**mod_par_monk3)

    # training parameters
    params_monk1 = {'lr': 0.8, 'momentum': 0.8, 'nesterov': True, 'epochs': 500, 'batch_size': 'full',
                    'error_func': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk2 = {'lr': 0.8, 'momentum': 0.8, 'nesterov': True, 'epochs': 500, 'batch_size': 'full',
                    'error_func': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk3_noreg = {'lr': 0.8, 'momentum': 0.8, 'nesterov': True, 'epochs': 500, 'batch_size': 'full',
                          'error_func': 'squared_error', 'metr': 'binary_class_accuracy'}

    params_monk3_l2 = {'lr': 0.76, 'momentum': 0.8, 'nesterov': True, 'epochs': 500, 'batch_size': 'full', 'lambda_': 0.005,
                       'reg_type': 'ridge_regression', 'error_func': 'squared_error', 'metr': 'binary_class_accuracy'}

    # Average prediction results
    avg_tr_error, avg_tr_acc, avg_ts_error, avg_ts_acc = [], [], [], []

    # model selection monks
    
    # kfold_CV(net=model_monk1, dataset='monks-1.train', **params_monk1, k_folds=5, verbose=True, plot=True,
    #        disable_tqdms=(False, False))

    # test prediction - 10 trials
    for trials in range(10):
        model_monk3 = Network(**mod_par_monk3)
        model_monk3.compile(**params_monk3_l2)
        tr_error_values, tr_acc_values, ts_error_values, ts_acc_values = model_monk3.fit(
            tr_x=monk_train3,
            tr_y=labels3,
            val_x=monk_test3,
            val_y=labels_ts3,
            disable_tqdm=False,
            epochs=500,
            batch_size='full'
        )
        # plot_curves(tr_error_values, ts_error_values, tr_acc_values, ts_acc_values, lbltr='Training', lblval='Test', ylim = (0., 1.1), ylim2 = (0., 1.1))
        avg_tr_error.append(tr_error_values[-1])
        avg_tr_acc.append(tr_acc_values[-1])
        avg_ts_error.append(ts_error_values[-1])
        avg_ts_acc.append(ts_acc_values[-1])

    # display average loss/accuracy training - test
    print('Average predictions')
    print(f"Train Loss: {np.mean(avg_tr_error)} - Test Loss: {np.mean(avg_ts_error)}\n"
          f"Train Acc: {np.mean(avg_tr_acc)} - Test Acc: {np.mean(avg_ts_acc)}")
    # ~ 27 seconds for one trial in monk-1
    # ~ 36 seconds for one trial in monk-2
    # ~ 18 seconds for one trial in monk-3 no reg
    # ~ 35 seconds for one trial in monk-3 reg