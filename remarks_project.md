## NEURAL NETWORK MODEL IMPLEMENTATION REMARKS
- Use 1-hot encoding for categories ✓
- DON'T USE TEST SET!!! ONLY FOR MODEL EVALUATION!! ✓
- Use LMS (divide by mb for mini-batch) as error metric (we may implement also other metrics like MEE). ✓
- Initialize weights by random values near zero (or initialization from the paper by Bengio and Glorot). ✓
- Try a random starting number of trials/configurations (initial weights). ✓
- Use mini-batch version (aka SGD) ✓
- Check the learning curve for learning rate ✓
- Use MOMENTUM (we may implement other type of momentum). ✓
- Stop training based on some criteria in every trial (NOT WITH A FIXED VALUES OF EPOCHS!!). ✓
- Use Tikhonov regularization (weights decay)!! (we may implement other kind of regularization). ✓
- Report only the error term in the report (plots or tables), not the entire loss! ✓
- Keep separations between lambda, momentum and eta in the implementation (slide 66 NOTE ON NN - part 2). ✓
- Use high number of units but with regularization. ✓
- Use sigmoid function for classification in output units with threshold (we may implement also softmax, cross entropy, etc.). ✓
- Use exhaustive grid-search to find best hyperparameters values (MODELS SELECTION).
Do it only for hyperparameters directly related to the VC-dim.
- Insert also STANDARD DEVIATION in the report, not only the MEAN of the error!! (K-FOLD CV with k=5 or 10...).
- Use CROSS-VALIDATION for training and validation set and then use a separate test set. ✓
- Compute mean error for training, validation and test set considering different trials (initialization of the weights). ✓
- Verify the model on different dataset (MONK + ML CUP). ✓
- Describe in the report how we get the last model from the validation phase! ✓
- Use MLP with backpropagation, momentum and L2 regularization. ✓
- Compare our simulator with an "oracle" tool to assess its correctness (Keras, Pytorch, etc.).
- Try to implement an efficient code for the experiments! ✓
- Implement some decay technique for learning rate ✓

## PYTHON IMPLEMENTATION REMARKS
- Use static methods for utility classes (when we don't care about the property of the object)
- Use @property decorator to implement setter and getter methods
-  
    - _var -> is meant as a hint to another programmer that a variable or method starting with a single underscore is intended for internal use.
    It should be considered an implementation detail and subject to change without notice.
    - __var -> this is also called name mangling; the interpreter changes the name of the variable in a way that makes it harder to create collisions when the class is extended later.

## Outline of things to implement
- Neural Network ✓
    - Activation functions, weights initialization ✓
    - Error functions, metrics, regularizations ✓
    - Learning rate decay ✓
    - Layer class ✓
        - Forward propagation (for one layer) ✓
        - Backpropagation (for one layer) ✓
    - Network class ✓
        - Feedforward propagation (entire network) ✓
        - Backpropagation (entire network) ✓
        - Network preparation (definition of error function,metric, regularizations, etc.) ✓
        - Training of the network (gradient descent) ✓
        - Prediction of the network ✓
        - Evaluation of the network with a target vector ✓
- Some trials for screening phase ✓
- Model Selection
    - Hold-out ✓
    - K-fold cross validation ✓
    - Grid Search ✓
    - Random Search ✓

## Screening Phase

MOMENTUM IN [0.5, 0.9] ✓
USE GLOROT FOR INITIALIZATION ✓


 """ HYPERPARAMETERS """
    units_per_layer = [20, 20, 2]
    act_functions = ['tanh', 'tanh','identity']
    weights_init = 'random' # or glorot
    error_func = 'squared_error'
    metr = 'euclidian_error'
    reg_type = 'ridge_regression'  # default regularization type is Tikhonov
    batch_size = 100 # try out batch sizes in powers of 2 (for better memory optimization) based on the data-size
    epochs = 500
    lr = 0.01
    momentum_val = 0.6
    lambda_val = 0
It seems that glorot and random perform similarly (does't matter initialization for larger net)
glorot e random indifferente (glorot sembra di pochissimo migliore ma hanno curve molto simili)
batch_size = 1 -> UNSTABLE LEARNING
batch_size = 'full' -> decrease very slowly the error
eta > 0.1 -> OVERFLOW
We have to use eta < 0.1
eta = 0.1 -> error doesn't decrease and remain on 20-21
eta = 0.05 -> unstable
eta = 0.001 -> very smooth curves
lambda = 0.1 overfitting pesissimo
momentum a 0.9 troppo alto (modello instabile)
more layers and no regularization -> not stable and overfitting
seems better one layer with very few regularization
unstable model and underfitting with few units (less than 10 units)
with ReLu or leaky_relu we have to set eta <= 0.0005, otherwise OVERFLOW (very similar learning curve for both, maybe better ReLU)
with ReLU curve very very smooth, stable but slower respect tanh
batch_size = 64 -> not bad results but a bit unstable
batch_size >= 512 -> very very smooth curves but error is not so good
limit_step <= 300 -> error is not good (linear_decay doesn't seem to improve learning)
we used eta = 0.001 so we discarded some decay because we had no improvements
if at epoch 50 validation error isn't below 30, DISCARD MODEL!
TODO -> add info on images of screening phase cup

## GRID SEARCH
### Fixed hyperparameters
weights_init = glorot
error_func = squared_error
metr = euclidian_error
reg_type = ridge_regression
nesterov = True

### Other hyperparameters
momentum = [0.6, 0.7, 0.8]
lambda_ = [0, 0.0001, 0.001]
units_per_layer =  [(10,2), (20, 2), (20, 20, 2), (20, 20, 20, 2)]
act_function = [('relu', 'identity'), ('tanh', 'identity'),
                          ('relu', 'relu', 'identity'), ('tanh', 'tanh', 'identity'),
                          ('relu', 'relu', 'relu', 'identity'),
                          ('tanh', 'tanh', 'tanh', 'identity')]
batch_size = [128, 256]
lr = [0.0001, 0.001, 0.005, 0.01]
reg_type = ['lasso','ridge_regression']
lr_decay = [None, 'linear', 'exponential']
epochs = 350
limit_step = 350 
decay_rate = 0.95 -> DA RIVEDERE
decay_steps = 350 

DILETTA
units_per_layer =  [(20, 2), (20, 20, 2)]
act_function = [('relu', 'identity'), ('tanh', 'identity'),
                          ('relu', 'relu', 'identity'), ('tanh', 'tanh', 'identity')]
PAOLO
units_per_layer =  [(10,2), (20, 20, 20, 2)]
act_function = [('relu', 'identity'), ('tanh', 'identity'),
                          ('relu', 'relu', 'relu', 'identity'),
                          ('tanh', 'tanh', 'tanh', 'identity')]


1) CHANGE INTERNAL TEST SIZE IN READ CUP ✓
2) SCREENING PHASE BATCH_SIZE ✓
4) SCREENING PHASE L1/L2 REGULARIZATION ✓ difficile capire il migliore, vanno messi entrambi nella grid
3) SCREENING PHASE LR_PARAMS
5) LEGGERE CODICE MODEL_SELECTION
6) AVVIARE GRID_SEARCH

sfoltire batch_size, learning rate e lambda
learning rate, lambda_, momentum and batch_size are perturbed in fine grid-search