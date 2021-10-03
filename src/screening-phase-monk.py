from net import Network
from utility import *
from sklearn.model_selection import train_test_split
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np

""" Exploring the effects of regularization techniques on learning """

if __name__ == '__main__':

    """ HYPERPARAMETERS """
    rescale=True # [-1, 1]
    units_per_layer = [2, 1]
    act_functions = ['tanh', 'tanh'] # use ['relu', 'sigmoid'] if rescale=False
    weights_init = 'random' # or glorot
    loss = 'squared_error'
    metr = 'binary_class_accuracy'
    # reg_type = 'lasso'  # default regularization type is Tikhonov
    batch_size = 'full' # try out batch sizes in powers of 2 (for better memory optimization) based on the data-size
    epochs = 300
    lr = 0.01
    momentum_val = [0, 0.7, 0.73, 0.77, 0.8, 0.85]
    lambda_val = [0, 0.0001, 0.0003, 0.0005, 0.0008, 0.001]


    # MONKS DEMO

    # Create a neural network
    # input_dim must stay 17 for monks datasets

    model = Network(input_dim=17, units_per_layer=units_per_layer, act_functions=act_functions, weights_init=weights_init,
                    tqdm=True)

    # read the dataset. Change the name in the following line to use monks-2 or monks-3
    tr_ds_name = "monks-1.train"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name, rescale=rescale)


    # hold-out validation
    train_X, val_X, train_y, val_y = train_test_split(monk_train, labels_tr, test_size=0.30)

    print(
        '\nHold-out validation. Shape check:'
        '\n Input TR: ', train_X.shape,
        '\n Output TR: ', val_X.shape,
        '\n Input VAL: ', train_y.shape,
        '\n Output VAL: ', val_y.shape,
    )

    # Plot Loss

    f, axs = plt.subplots(nrows=6, ncols=6, figsize=(36, 36))
    plot_id = 0

    for i in lambda_val:
        for j in momentum_val:
            model.compile(error_func=loss, metr=metr, lr=lr, momentum=j, lambda_=i)

            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y,
                                                           batch_size=batch_size,
                                                           epochs=epochs, tqdm=True)
            # to plot the loss curve: substitute tr_metr with tr_err and val_metr with val_err
            axs[int(plot_id / 6)][plot_id % 6].plot(range(len(tr_metr)), tr_metr, color='b', linestyle='dashed', label='Training')
            axs[int(plot_id / 6)][plot_id % 6].plot(range(len(val_metr)), val_metr, color='r', label='Validation')
            axs[int(plot_id / 6)][plot_id % 6].legend(loc='best', prop={'size': 9})
            axs[int(plot_id / 6)][plot_id % 6].set_xlabel('Epochs', fontweight='bold')
            axs[int(plot_id / 6)][plot_id % 6].set_ylabel('Accuracy', fontweight='bold') # change to 'Error' if plotting error curve
            # axs[int(plot_id / 6)][plot_id % 6].set_ylim([0, 0.5])
            axs[int(plot_id / 6)][plot_id % 6].set_title(f'lambda: {str(i)}, momentum: {str(j)}')
            axs[int(plot_id / 6)][plot_id % 6].grid()
            plot_id = plot_id + 1

    plt.suptitle(f'Momentum and Lambda variations with'
                 f' learning_rate = {str(lr)}, units_per_layer = {str(units_per_layer)}, act_functions = {str(act_functions)}, weights_init = {str(weights_init)}, \n loss = {str(loss)}, metr = {str(metr)}, batch_size = {str(batch_size)}, epochs = {str(epochs)}', fontsize=28, fontweight='bold')
    plt.savefig(f'../images/acc7.png')

