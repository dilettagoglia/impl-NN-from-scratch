from net import Network
from utility import *
from sklearn.model_selection import train_test_split
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt
import numpy as np



if __name__ == '__main__':
    # MONKS DEMO

    # Create a neural network
    # input_dim must stay 17 for monks datasets
    # units_per_layer: tuple containing the number of units for each layer (except the input one)
    model = Network(input_dim=17, units_per_layer=[4, 1], act_functions=['tanh', 'tanh'], weights_init='random',
                    tqdm=True)

    # read the dataset. Change the name in the following lines to use monks-2 or monks-3
    tr_ds_name = "monks-1.train"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name)
    # monk_test, labels_ts = read_monk_dataset(dataset="monks-1.test")

    # Validation alternatives:
    # NOTE: do not consider the following hyperparameters as hints, they were put down very quickly.
    # NOTE: keep not commented only one alternative

    # hold-out validation
    # # compile the model (check the method definition for more info about all the accepted arguments)
    train_X, val_X, train_y, val_y = train_test_split(monk_train, labels_tr, test_size=0.30)

    print(
        '\nHold-out validation. Shape check:'
        '\n Input TR: ', train_X.shape,
        '\n Output TR: ', val_X.shape,
        '\n Input VAL: ', train_y.shape,
        '\n Output VAL: ', val_y.shape,
    )


    """ HYPERPARAMETER TUNING """
    lr_rates = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    momentum_val = [0, 0.01, 0.015, 0.02, 0.25, 0.03]
    lambda_val = [0, 0.01, 0.015, 0.02, 0.25, 0.03]

    # # plot the learning curves

    f, axs = plt.subplots(nrows=6, ncols=6, figsize=(36, 36))
    plt.suptitle('Momentum and Eta variations with (lambda=0, 4 hidden units, and random weight init [-1,+1]', fontsize=28, fontweight='bold')
    plot_id = 0

    for i in lr_rates:
        for j in momentum_val:
            model.compile(loss='squared_error', metr='binary_accuracy', lr=i, momentum=j)
            # lr_decay
            # limit_step
            # decay_rate (float, optional): amount of decay at each stage (for exponential). Defaults to None.
            # decay_steps (int, optional): length of each stage for decaying, composed of multiple iterations (steps). Defaults to None.
            # momentum (float, optional): momentum parameter. Defaults to 0..
            # reg_type (str, optional): regularization type. Defaults to 'ridge_regression'.
            # lambda_ (int, optional): regularization parameter. Defaults to 0.

            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=train_X, tr_y=train_y, val_x=val_X, val_y=val_y,
                                                           batch_size='full',
                                                           epochs=300, tqdm=True)

            axs[int(plot_id / 6)][plot_id % 6].plot(range(len(tr_err)), tr_err, color='b', linestyle='dashed', label='Training')
            axs[int(plot_id / 6)][plot_id % 6].plot(range(len(val_err)), val_err, color='r', label='Validation')
            axs[int(plot_id / 6)][plot_id % 6].legend(loc='best', prop={'size': 9})
            axs[int(plot_id / 6)][plot_id % 6].set_xlabel('Epochs', fontweight='bold')
            axs[int(plot_id / 6)][plot_id % 6].set_ylabel('Error', fontweight='bold')
            # axs[int(plot_id / 6)][plot_id % 6].set_ylim([0, 0.5])
            axs[int(plot_id / 6)][plot_id % 6].set_title(f'eta: {str(i)}, momentum: {str(j)}')
            axs[int(plot_id / 6)][plot_id % 6].grid()
            plot_id = plot_id + 1

    plt.savefig(f'../images/momentum-and-learning-rate-variations3.png')

