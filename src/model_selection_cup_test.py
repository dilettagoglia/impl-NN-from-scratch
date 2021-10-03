from utility import read_cup
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # grid search parameters
    gs_params = {'units_per_layer': ([20, 2], [20, 20, 2]),
                 'act_functions': (['leaky_relu', 'identity'], ['tanh', 'identity'],
                          ['leaky_relu', 'leaky_relu', 'identity'], ['tanh', 'tanh', 'identity'],),
                 'weights_init': ('random', 'glorot',),
                 'bounds': ((-0.1, 0.1), (-0.001, 0.001)),
                 'momentum': (0.5, 0.8),
                 'batch_size': (1, 200, 'full'),
                 'lr': (0.001, 0.0001),
                 'lr_decay': (None, 'linear_decay', 'exponential_decay'),
                 'limit_step': (400,),
                 'decay_rate': (0.95,),
                 'decay_steps': (400,),
                 'lambda_': (0, 0.001, 0.0001, 0.00001),
                 'reg_type': ('lasso', 'ridge_regression'),
                 'error_func': ('squared_error',),
                 'metr': ('binary_class_accuracy',),
                 'epochs': (150, 400, 700)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset="cup", params=gs_params, coarse=True)
    _, best_params = get_best_models(dataset="cup", coarse=True, n_models=5)
    #best_params = best_params[0]
    #grid_search(dataset="cup", params=best_params, coarse=False, n_config=4)
    #best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=10)
    for p in best_params:
        print(p)