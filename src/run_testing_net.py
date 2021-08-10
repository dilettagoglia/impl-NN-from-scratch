from net import Network
from utility import *
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == '__main__':
    # MUSCA TEST
    # MONKS DEMO

    # Create a neural network
    # input_dim must stay 17 for monks datasets
    # units_per_layer: tuple containing the number of units for each layer (except the input one)
    model = Network(input_dim=17, units_per_layer=[16, 2], act_functions=['tanh', 'tanh'], weights_init='random', tqdm=True)

    # read the dataset. Change the name in the following lines to use monks-2 or monks-3
    tr_ds_name = "monks-1.train"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name)
    #monk_test, labels_ts = read_monk_dataset(dataset="monks-1.test")

    # Validation alternatives:
    # NOTE: do not consider the following hyperparameters as hints, they were put down very quickly.
    # NOTE: keep not commented only one alternative

    # hold-out validation
    # # compile the model (check the method definition for more info about all the accepted arguments)
    train_X, val_X, train_y, val_y = train_test_split(monk_train, labels_tr, test_size=0.30)
    model.compile(loss='squared_error', metr='binary_accuracy', lr=0.3, momentum=0.8, nesterov=True, lambda_=0.01)
    # # training (check the method definition for more info about all the possible parameters)
    tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y, batch_size='full',
                                                    epochs=500)
    # # plot the learning curves
    plot_curves(tr_err, val_err, tr_metr, val_metr, plt_title="Test net", lbltr='Training', lblval='Validation')



