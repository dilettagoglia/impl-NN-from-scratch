from utility import read_monk_dataset
from model_selection import grid_search, get_best_models

if __name__ == '__main__':
    # read dataset
    monk_train1, labels1 = read_monk_dataset('monks-1.train', rescale=False)

    # grid search parameters
    gs_params = {'units_per_layer': ((20, 2), (20, 20, 2), (20, 20, 10, 2), (8, 8, 8, 8, 8, 2)),
                 'act_functions': (('leaky_relu', 'identity'), ('tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'identity'), ('tanh', 'tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'leaky_relu', 'identity'),
                          ('tanh', 'tanh', 'tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'identity'),
                          ('tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'identity'),),
                 'weights_init': ('random','glorot'),
                 'limits': ((-0.1, 0.1), (-0.001, 0.001)),
                 'momentum': (0.5, 0.8),
                 'batch_size': (1, 200, 'full'),
                 'lr': (0.001, 0.0001),
                 'lr_decay': (None, 'linear', 'exponential'),
                 'limit_step': (400,),
                 'decay_rate': (0.95,),
                 'decay_steps': (400,),
                 'lambd': (0, 0.001, 0.0001, 0.00001),
                 'reg_type': ('l1', 'l2'),
                 'staircase': (True, False),
                 'loss': ('squared',),
                 'metr': ('euclidean',),
                 'epochs': (150, 400, 700)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset=monk_train1, params=gs_params, coarse=True)
    _, best_params = get_best_models(dataset=monk_train1, coarse=True, n_models=5)
    best_params = best_params[0]
    grid_search(dataset=monk_train1, params=best_params, coarse=False, n_config=4)
    best_models, best_params = get_best_models(dataset=monk_train1, coarse=False, n_models=10)
    for p in best_params:
        print(p)