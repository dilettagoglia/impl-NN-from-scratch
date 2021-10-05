from net import Network
from utility import *
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt

""" Exploring the effects of regularization techniques on learning """

if __name__ == '__main__':

    """ HYPERPARAMETERS """
    units_per_layer = [20, 2]
    act_functions = ['leaky_relu', 'identity']
    weights_init = 'random' # or glorot
    error_func = 'squared_error'
    metr = 'euclidian_error'
    reg_type = 'lasso'  # default regularization type is Tikhonov
    batch_size = 'full' # try out batch sizes in powers of 2 (for better memory optimization) based on the data-size
    epochs = 200
    lr = 0.7
    momentum_val = [0.5, 0.7, 0.8, 0.9]
    lambda_val = [0, 0.001, 0.0001, 0.00001]

    # input dimension for cup is 10
    model = Network(input_dim=10, units_per_layer=units_per_layer, act_functions=act_functions, weights_init=weights_init,
                    tqdm=True)

    tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data = read_cup(int_ts=True) # detach internal test set

    print(int_ts_data)
    print(int_ts_targets)

    # Plot Loss

    f, axs = plt.subplots(nrows=4, ncols=4, figsize=(36, 36))
    plot_id = 0

    for i in lambda_val:
        for j in momentum_val:
            print('compilo', i, j) # debugging print
            model.compile(error_func=error_func, metr=metr, lr=lr, momentum=j, lambda_=i)

            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_data, tr_targets, val_x=int_ts_data, val_y=int_ts_targets,
                                                           batch_size=batch_size,
                                                           epochs=epochs, tqdm=True)
            plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation')
            '''
            # to plot the loss curve: substitute tr_metr with tr_err and val_metr with val_err
            axs[int(plot_id / 4)][plot_id % 4].plot(range(len(tr_metr)), tr_metr, color='b', linestyle='dashed', label='Training')
            axs[int(plot_id / 4)][plot_id % 4].plot(range(len(val_metr)), val_metr, color='r', label='Validation')
            axs[int(plot_id / 4)][plot_id % 4].legend(loc='best', prop={'size': 9})
            axs[int(plot_id / 4)][plot_id % 4].set_xlabel('Epochs', fontweight='bold')
            axs[int(plot_id / 4)][plot_id % 4].set_ylabel('Accuracy', fontweight='bold') # change to 'Error' if plotting error curve
            axs[int(plot_id / 4)][plot_id % 4].set_ylim([0, 0.5])
            axs[int(plot_id / 4)][plot_id % 4].set_title(f'lambda: {str(i)}, momentum: {str(j)}')
            axs[int(plot_id / 4)][plot_id % 4].grid()
            plot_id = plot_id + 1
            

    plt.suptitle(f'Momentum and Lambda variations with'
                 f' learning_rate = {str(lr)}, units_per_layer = {str(units_per_layer)}, act_functions = {str(act_functions)}, weights_init = {str(weights_init)}, \n loss = {str(error_func)}, metr = {str(metr)}, batch_size = {str(batch_size)}, epochs = {str(epochs)}', fontsize=28, fontweight='bold')
    plt.savefig(f'../images/cup_screening_phase.png')
    '''
