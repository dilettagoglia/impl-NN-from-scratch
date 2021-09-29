from net import Network
from utility import *
from sklearn.model_selection import train_test_split
import numpy as np
from model_selection import kfold_CV


if __name__ == '__main__':
    # MONKS DEMO

    # Create a neural network
    # input_dim must stay 17 for monks datasets
    # units_per_layer: tuple containing the number of units for each layer (except the input one)
    model = Network(input_dim=17, units_per_layer=[2, 1], act_functions=['tanh', 'tanh'], weights_init='glorot', tqdm=True)

    # read the dataset. Change the name in the following lines to use monks-2 or monks-3
    tr_ds_name = "monks-1.train"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name, rescale=True, preliminary_analysis=False)
    #monk_test, labels_ts = read_monk_dataset(dataset="monks-1.test")

    '''********* hold-out validation ************** '''
    #train_X, val_X, train_y, val_y = train_test_split(monk_train, labels_tr, test_size=0.30)

    #model.compile(error_func='squared_error', metr='binary_class_accuracy', lr=0.07, momentum=0.7, lambda_=0.0001)

    #tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y, batch_size='full', epochs=150, tqdm=True)

    ''' ********** cross validation ***************** '''
    #kfold_CV(net=model, dataset=tr_ds_name, loss='squared_error', metr='binary_class_accuracy', lr=0.76, opt='sgd', momentum=0.83,
                #epochs = 500, batch_size = 'full', k_folds = 5, disable_tqdms = (True, False), plot = True, verbose = True)


    # # plot the learning curves
    # plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation')

    # model.print_topology()



