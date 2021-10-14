from model_selection import *
from net import Network
from utility import read_cup, save_blind
import numpy as np

if __name__ == '__main__':

    # read cup {development set - internal test set}
    dev_set_x, dev_set_labels, int_ts_x, int_ts_labels, _ = read_cup(int_ts=True)
    MODE = 'prediction'
    # MODEL PARAMETERS
    units_per_layer = [20, 20, 2]
    act_functions = ['relu', 'relu', 'identity']

    # model parameters
    mod_par_cup = {
        'input_dim': 10,
        'units_per_layer': units_per_layer,
        'act_functions': act_functions,
        'weights_init': 'glorot'
    }

    # TRAINING PARAMETERS
    lr = 0.001
    lr_decay = None # or 'linear_decay'
    limit_step = None
    momentum = 0.6
    lambda_ = 0
    reg_type = 'ridge_regression'
    batch_size = 145

    # training parameters dict
    params_cup = {'lr': lr, 'lr_decay': lr_decay, 'limit_step': limit_step, 'momentum': momentum, 'nesterov': True, 'epochs': 350, 'batch_size': batch_size,
                    'error_func': 'squared_error', 'metr': 'euclidian_error', 'lambda_': lambda_}

    # MODEL SELECTION
    if MODE == 'validation':
        print("----- MODEL SELECTION -----")
        model_cup = Network(**mod_par_cup)
        # model selection cup
        # with plot=True it will plot a graph for each fold
        kfold_CV(net=model_cup, dataset='cup', **params_cup, k_folds=5, verbose=True, plot=True,
                disable_tqdms=(False, False))

    # MODEL ASSESSMENT ON INTERNAL TEST SET
    elif MODE == 'int_ts_assessment':
        print("----- MODEL ASSESSMENT ON INTERNAL TEST SET -----")
        # Average prediction results
        avg_tr_error, avg_tr_acc, avg_ts_error, avg_ts_acc = [], [], [], []
        # test prediction - 10 trials
        for trials in range(10):
            model_cup = Network(**mod_par_cup)
            model_cup.compile(**params_cup)
            tr_error_values, tr_acc_values, ts_error_values, ts_acc_values = model_cup.fit(
                tr_x=dev_set_x,
                tr_y=dev_set_labels,
                val_x=int_ts_x,
                val_y=int_ts_labels,
                disable_tqdm=False,
                epochs=350,
                batch_size= batch_size
            )
            # with path=None plot_curves will plot a graph for each trial. If we want only save images pass path != None to plot_curves 
            path = None # path = '../images/int_ts_assessment/model_assessment_{}'.format(trials)
            plot_curves(tr_error_values, ts_error_values, tr_acc_values, ts_acc_values, ylim2 = (0., 20.), path=path)
            avg_tr_error.append(tr_error_values[-1])
            avg_tr_acc.append(tr_acc_values[-1])
            avg_ts_error.append(ts_error_values[-1])
            avg_ts_acc.append(ts_acc_values[-1])

        # display average loss/accuracy training - test
        print('Average predictions')
        print(f"Train Loss: {np.mean(avg_tr_error)} - Test Loss: {np.mean(avg_ts_error)}\n"
            f"Train Acc: {np.mean(avg_tr_acc)} - Test Acc: {np.mean(avg_ts_acc)}") # keep this for our Table 3 (dev set MEE and internal test set MEE)
    
    # PREDICTION ON BLIND TEST SET
    else:
        print("----- PREDICTION ON BLIND TEST SET -----")
        tr_data, tr_targets, cup_ts_data = read_cup()
        model_cup = Network(**mod_par_cup)
        model_cup.compile(**params_cup)
        tr_error_values, tr_acc_values, ts_error_values, ts_acc_values = model_cup.fit(
            tr_x=tr_data,
            tr_y=tr_targets,
            disable_tqdm=False,
            epochs=350,
            batch_size= batch_size
        )
        save_blind(model_cup, cup_ts_data)        