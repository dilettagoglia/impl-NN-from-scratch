import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
from utility import *
from net import *
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics

# Monk train dataset analysis
ds_names=['monks-1.train', 'monks-2.train', 'monks-3.train']
for dataset in ds_names:
    print('\n\n **** Analysis of "',dataset,'" dataset ****')
    monk_train, labels_tr = read_monk_dataset(dataset, preliminary_analysis=True)
    print('One-Hot encoded dataset. \n Training set:\n', monk_train, '\n Labels:\n', labels_tr.T)
    # Check dimensions
    print('\nLabels vector shape:\n', labels_tr.shape, '\n Training dataset shape: (examples, features)\n', monk_train.shape)

# Cup train dataset analysis
print('\n\n **** Analysis of ML-CUP20-TR dataset ****')
tr_data, tr_targets, int_ts_data, int_ts_targets, cup_ts_data = read_cup(int_ts=True)
# Check dimensions
print('\nLabels vector shape:\n', tr_targets.shape, '\n Training dataset shape: (examples, features)\n', tr_data.shape)


''' BIAS VARIANCE DECOMPOSITION'''
epochs = 250
batch_size = 145

devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)
# Model definition
model = Network(input_dim=10, units_per_layer=[20, 20,  2], act_functions=['relu', 'relu', 'identity'], weights_init='glorot')
model.compile(error_func='squared_error', metr='euclidian_error', lr=0.001, lr_decay=None, momentum=0.6, nesterov=True, lambda_=0, reg_type='ridge_regression')

# Estimation of bias and variance using bias_variance_decomp
def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]

def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test, num_rounds=50, random_seed=None, **fit_params):
    rng = np.random.RandomState(random_seed)
    all_pred = []

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)

        pred = estimator.fit(X_boot, y_boot, **fit_params).predict(X_test)
        all_pred.append(pred)

    main_predictions = np.mean(all_pred, axis=0)

    avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size
    avg_var = np.sum((main_predictions - all_pred) ** 2) / len(all_pred)

    return avg_bias, avg_var


# Note here we are using loss as 'mse' and setting default bootstrap num_rounds to 200
bias, var = bias_variance_decomp(model, devset_x, devset_y, int_ts_x, int_ts_y, num_rounds=50, random_seed=123,
                                      epochs=epochs, batch_size=batch_size, error_exp=True)

y_pred = model.predict(int_ts_x)
# summarize results
print('Avg Bias: %.3f' % bias)
print('Avg Variance: %.3f' % var)
print('Mean Square error by Sckit-learn lib: %.3f' % metrics.mean_squared_error(int_ts_y, y_pred))


'''
RESULTS

(latest run: 14 oct 2021)

Avg Bias: 7.353
Avg Variance: 325.097
Mean Square error by Sckit-learn lib: 9.317
'''