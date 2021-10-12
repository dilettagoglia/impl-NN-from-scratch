from utility import read_cup
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # grid search parameters
    gs_params = {'units_per_layer': ([10,2],[20,2],[20,20,2]),
                 'act_functions': ( ['relu', 'identity'], ['tanh', 'identity'],
                                    ['relu', 'relu', 'identity'], ['tanh', 'tanh', 'identity']),
                 'weights_init': ('glorot',),
                 'momentum': (0.6, 0.7, 0.8),
                 'nesterov': (True,),
                 'batch_size': (128, 256),
                 'lr': (0.0001, 0.001 ,0.01),
                 'lr_decay': (None, 'linear_decay'),
                 'limit_step': (350,),
                 'lambda_': (0, 0.0001, 0.001),
                 'reg_type': ('lasso','ridge_regression'),
                 'error_func': ('squared_error',),
                 'metr': ('euclidian_error',),
                 'epochs': (350,)}
    baseline_es = {'epoch': 50, 'threshold': 25}

    # coarse to fine grid search. Results are saved on file

    ''' COARSE'''
    # grid_search(dataset="cup", params=gs_params, coarse=True, baseline_es=baseline_es)

    _, best_params = get_best_models(dataset="cup", coarse=True, n_models=3, fn='coarse_gs_results_cup_diletta.json')
    best_params = best_params[0]
    #for p, v in best_params.items():
    #    print(p, v)

    ''' FINE '''
    grid_search(dataset="cup", params=best_params, coarse=False, n_config=3, baseline_es=baseline_es)
    best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=5, fn='fine_gs_results_cup_diletta_2.json')
    #print(best_models)

