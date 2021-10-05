from utility import read_monk_dataset
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    tr_ds_name = "monks-3.train"
    ts_ds_name = "monks-3.test"

    # grid search parameters
    gs_params = {'units_per_layer': ([4, 1], [8, 1], [16, 1]),
                 'act_functions': (['relu', 'sigmoid'],),
                 'weights_init': ('glorot',),
                 'momentum': (0.8, 0.9),
                 'nesterov': (True,),
                 'batch_size': ('full',),
                 'lr': (0.76, 0.8),
                 'error_func': ('squared_error',),
                 'metr': ('binary_class_accuracy',),
                 'lambda_': (0.001, 0.0001, 0.005),
                 'reg_type': ('lasso', 'ridge_regression'),
                 'epochs': (500,)}

    # coarse to fine grid search. Results are saved on file
    # grid_search(dataset=tr_ds_name, params=gs_params, coarse=True)
    _, best_params = get_best_models(dataset=tr_ds_name, coarse=True, n_models=5)
    best_params = best_params[0]
    # grid_search(dataset=tr_ds_name, params=best_params, coarse=False, n_config=4)
    # best_models, best_params = get_best_models(dataset=tr_ds_name, coarse=False, n_models=5)
    for p,v in best_params.items():
        print(p,v)