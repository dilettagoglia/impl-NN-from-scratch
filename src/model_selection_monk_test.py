from utility import read_monk_dataset
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    tr_ds_name = "monks-1.train"
    monk_train, labels_tr = read_monk_dataset(dataset=tr_ds_name, rescale=False)

    #TODO change name for paramaters to adapt to our network ('error_func' instead of 'loss', etc. ) -> check net.py file
    # grid search parameters
    gs_params = {'units_per_layer': ([2, 2], [4, 2], [3, 2]),
                 'act_functions': (['leaky_relu', 'sigmoid'],),
                 'weights_init': ('random', 'glorot',),
                 #'bounds': ((-0.1, 0.1), (-0.001, 0.001)),
                 'momentum': (0.5, 0.8),
                 'batch_size': (1, 200, 'full'),
                 'lr': (0.001, 0.0001),
                 'lr_decay': (None, 'linear_decay', 'exponential_decay'),
                 'limit_step': (400,),
                 'decay_rate': (0.95,),
                 'decay_steps': (400,),
                 'lambda_': (0, 0.001, 0.0001, 0.00001),
                 'reg_type': ('lasso', 'ridge_regression'),
                 #'staircase': (True, False),
                 'error_func': ('squared_error',),
                 'metr': ('binary_class_accuracy',),
                 'epochs': (150, 400, 700)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset=tr_ds_name, params=gs_params, coarse=True)
    _, best_params = get_best_models(dataset=tr_ds_name, coarse=True, n_models=5)
    best_params = best_params[0]
    grid_search(dataset=tr_ds_name, params=best_params, coarse=False, n_config=4)
    best_models, best_params = get_best_models(dataset=tr_ds_name, coarse=False, n_models=5)
    for p in best_params:
        print(p)