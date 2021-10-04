from utility import read_monk_dataset
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    tr_ds_name = "monks-1.train"
    ts_ds_name = "monks-1.test"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name, rescale=False)
    monk_test, labels_ts = read_monk_dataset(dataset=ts_ds_name, rescale=False)

    # grid search parameters
    gs_params = {'units_per_layer': ([2, 1], [3, 1], [4, 1]),
                 'act_functions': (['relu', 'sigmoid'],),
                 'weights_init': ('random', 'glorot'),
                 'bounds': ((-0.25, 0.25),(-0.1, 0.1), (-0.01, 0.01)),
                 'momentum': (0.5, 0.8),
                 'nesterov': (True,),
                 'batch_size': ('full',),
                 'lr': (0.001, 0.0001),
                 'lambda_': (0, 0.001, 0.0001, 0.00001),
                 'error_func': ('squared_error',),
                 'metr': ('binary_class_accuracy',),
                 'epochs': (500,)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset=tr_ds_name, params=gs_params, coarse=True)
    """_, best_params = get_best_models(dataset=tr_ds_name, coarse=True, n_models=5)
    best_params = best_params[0]
    grid_search(dataset=tr_ds_name, params=best_params, coarse=False, n_config=4)
    best_models, best_params = get_best_models(dataset=tr_ds_name, coarse=False, n_models=5)
    for p in best_params:
        print(p)"""