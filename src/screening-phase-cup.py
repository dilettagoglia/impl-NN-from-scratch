from net import Network
from utility import *
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from matplotlib import pyplot as plt

""" Exploring the effects of regularization techniques on learning """

# TODO change based on variations for screening phase

if __name__ == '__main__':

    """ HYPERPARAMETERS """
    units_per_layer_val = [[20,2],[20, 20, 2], [20,20,20,2], [20,20,20,20,2]]
    act_functions_val = [['tanh','identity'],['tanh', 'tanh','identity'], ['tanh', 'tanh','tanh','identity'], ['tanh', 'tanh','tanh','tanh','identity']]
    weights_init = 'glorot' # or glorot
    error_func = 'squared_error'
    metr = 'euclidian_error'
    reg_type = 'ridge_regression'  # default regularization type is Tikhonov
    batch_size = 128 # try out batch sizes in powers of 2 (for better memory optimization) based on the data-size
    epochs = 250
    lr = 0.01
    # lr_val = [0.001, 0.005, 0.01, 0.05]
    momentum = 0.6
    # momentum_val = [0.6, 0.7, 0.8, 0.9]
    lambda_val = [0, 0.0001, 0.001, 0.1]
    # lambda_ = 0.0001

    # input dimension for cup is 10
    # model = Network(input_dim=10, units_per_layer=units_per_layer, act_functions=act_functions, weights_init=weights_init)

    tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data = read_cup(int_ts=True) # detach internal test set

    # print(int_ts_data)
    # print(int_ts_targets)

    # Plot Loss

    f, axs = plt.subplots(nrows=4, ncols=4, figsize=(36, 36))
    plot_id = 0

    for i,z in zip(units_per_layer_val, act_functions_val): 
        for j in lambda_val:
            model = Network(input_dim=10, units_per_layer=i, act_functions=z, weights_init=weights_init)
            model.compile(error_func=error_func, metr=metr, lr=lr, momentum=momentum, nesterov=True, lambda_=j)

            # # training (check the method definition for more info about all the possible parameters)
            tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=tr_data, tr_y=tr_targets, val_x=int_ts_data, val_y=int_ts_targets,
                                                           batch_size=batch_size,
                                                           epochs=epochs, disable_tqdm=False)
            ylim = 30
            if tr_err[50] > ylim or val_err[50] > ylim:
                ylim = 100
            # plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation',ylim=ylim)
            
            # to plot the loss curve: substitute tr_metr with tr_err and val_metr with val_err
            axs[int(plot_id / 4)][plot_id % 4].plot(range(len(tr_err)), tr_err, color='b', linestyle='dashed', label='Training')
            axs[int(plot_id / 4)][plot_id % 4].plot(range(len(val_err)), val_err, color='r', label='Validation')
            axs[int(plot_id / 4)][plot_id % 4].legend(loc='best', prop={'size': 9})
            axs[int(plot_id / 4)][plot_id % 4].set_xlabel('Epochs', fontweight='bold')
            axs[int(plot_id / 4)][plot_id % 4].set_ylabel('Error (MSE)', fontweight='bold') # change to 'Error' if plotting error curve
            axs[int(plot_id / 4)][plot_id % 4].set_ylim([0, ylim])
            axs[int(plot_id / 4)][plot_id % 4].set_title(f'units_per_layer: {str(i)}, lambda: {str(j)}')
            axs[int(plot_id / 4)][plot_id % 4].grid()
            plot_id = plot_id + 1
            

    plt.suptitle(f'Units per layer and lambda variations with'
                 f' momentum = {str(momentum)}, learning_rate = {str(lr)}, act_functions = tanh for hidden layer, weights_init = {str(weights_init)}, \n loss = {str(error_func)}, metr = {str(metr)}, batch_size = {str(batch_size)}, epochs = {str(epochs)}', fontsize=28, fontweight='bold')
    plt.savefig(f'../images/cup_screening_phase_cmp_unitslayer_lambda.png')
