from net import Network
from model_selection import *
from utility import *
from sklearn.model_selection import train_test_split
import numpy as np

# MUSCA TEST
if __name__ == '__main__':
    DATASET = "monk"
    if DATASET == "monk":
        # MONKS DEMO

        # Create a neural network
        # input_dim must stay 17 for monks datasets
        # units_per_layer: tuple containing the number of units for each layer (except the input one)
        model = Network(input_dim=17, units_per_layer=[4, 1], act_functions=['relu', 'sigmoid'], weights_init='random')
        # read the dataset. Change the name in the following lines to use monks-2 or monks-3
        tr_ds_name = "monks-2.train"
        ts_ds_name = "monks-2.test"
        VALIDATION = "we"
        if VALIDATION == "holdout_sklearn":
            monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name)
            #monk_test, labels_ts = read_monk_dataset(dataset="monks-1.test")

            # Validation alternatives:
            # NOTE: do not consider the following hyperparameters as hints, they were put down very quickly.
            # NOTE: keep not commented only one alternative

            # hold-out validation
            # # compile the model (check the method definition for more info about all the accepted arguments)
            train_X, val_X, train_y, val_y = train_test_split(monk_train, labels_tr, test_size=0.30)
            model.compile(error_func='squared_error', metr='binary_class_accuracy', lr=0.76, momentum=0.83, lambda_=0)
            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y, batch_size='full',
                                                            epochs=500, disable_tqdm=False)
            # # plot the learning curves
            plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation')
        elif VALIDATION == "holdout":
            tr_err, tr_metr, val_err, val_metr = holdout_validation(net=model, path=tr_ds_name, test_size=0.30, error_func='squared_error',
                                                metr='binary_class_accuracy', lr=0.76, momentum=0.83, lambda_=0,batch_size='full',
                                                            epochs=500, disable_tqdm=False)
            # # plot the learning curves
            plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation')
        elif VALIDATION == "kfold":
            kfold_CV(net=model, dataset=tr_ds_name, error_func='squared_error',
                                                metr='binary_class_accuracy', lr=0.76, momentum=0.83, lambda_=0,batch_size='full',
                                                            epochs=500, disable_tqdms=(False,False), verbose=True)
        else: # NO VALIDATION
            monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name)
            monk_test, labels_ts = read_monk_dataset(dataset=ts_ds_name)
            model.compile(error_func='squared_error', metr='binary_class_accuracy', lr=0.76, momentum=0.83, lambda_=0, nesterov=True)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=monk_train, tr_y=labels_tr, val_x=monk_test, val_y=labels_ts, batch_size='full',
                                                            epochs=500, disable_tqdm=False,)
            plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Test')
            print(tr_err[-1], tr_metr[-1])
            print(model.evaluate(targets=labels_ts, metr='binary_class_accuracy', error_func='squared_error', inp=monk_test))

    # CUP TESTS
    else:
        # Create a neural network
        # input_dim must stay 10 for cup datasets
        # units_per_layer: tuple containing the number of units for each layer (except the input one)
        model = Network(input_dim=10, units_per_layer=[20, 2], act_functions=['tanh', 'identity'], weights_init='glorot')
        # read the dataset. Change the name in the following lines to use monks-2 or monks-3
        tr_ds_name = "ML-CUP20-TR.csv"
        # ts_ds_name = "monks-3.test"
        VALIDATION = "holdout_sklearn"
        if VALIDATION == "holdout_sklearn":
            cup_train, labels_tr, _ = read_cup()

            # Validation alternatives:
            # NOTE: do not consider the following hyperparameters as hints, they were put down very quickly.
            # NOTE: keep not commented only one alternative

            # hold-out validation
            # # compile the model (check the method definition for more info about all the accepted arguments)
            train_X, val_X, train_y, val_y = train_test_split(cup_train, labels_tr, test_size=0.1)
            model.compile(error_func='squared_error', metr='euclidian_error', lr=0.002143, momentum=0.6, lambda_=0)
            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y, batch_size=16,
                                                            epochs=500, disable_tqdm=False)
            # # plot the learning curves
            ylim = (0,10)
            plot_curves(tr_err, val_err, tr_metr, val_metr,ylim=ylim)

    




