from utility import read_cup
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # grid search parameters
    gs_params = {'units_per_layer': ([10,2], [20, 2], [20, 20, 2], [20, 20, 20, 2]),
                 'act_functions': (['relu', 'identity'], ['tanh', 'identity'],
                          ['relu', 'relu', 'identity'], ['tanh', 'tanh', 'identity'],
                          ['relu', 'relu', 'relu', 'identity'],
                          ['tanh', 'tanh', 'tanh', 'identity']),
                 'weights_init': ('glorot',),
                 'momentum': (0.6, 0,7, 0.8),
                 'batch_size': (128, 256),
                 'lr': (0.0001, 0.001, 0.005,0.01),
                 'lr_decay': (None, 'linear_decay'),
                 'limit_step': (350,),
                 'lambda_': (0, 0.0001, 0.001),
                 'reg_type': ('lasso', 'ridge_regression'),
                 'error_func': ('squared_error',),
                 'metr': ('euclidian_error',),
                 'epochs': (350,)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset="cup", params=gs_params, coarse=True)
    _, best_params = get_best_models(dataset="cup", coarse=True, n_models=5)
    #best_params = best_params[0]
    #grid_search(dataset="cup", params=best_params, coarse=False, n_config=4)
    #best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=10)
    #for p in best_params:
    #    print(p)