## NEURAL NETWORK MODEL IMPLEMENTATION REMARKS
- Use 1-hot encoding for categories ✓
- DON'T USE TEST SET!!! ONLY FOR MODEL EVALUATION!!
- Use LMS (divide by mb for mini-batch) as error metric (we may implement also other metrics like MEE). ✓
- Initialize weights by random values near zero (or initialization from the paper by Bengio and Glorot). ✓
- Try a random starting number of trials/configurations (initial weights).
- Use mini-batch version (aka SGD) ✓
- Check the learning curve for learning rate ✓
- Use MOMENTUM (we may implement other type of momentum). ✓
- Stop training based on some criteria in every trial (NOT WITH A FIXED VALUES OF EPOCHS!!). ✓
- Use Tikhonov regularization (weights decay)!! (we may implement other kind of regularization). ✓
- Report only the error term in the report (plots or tables), not the entire loss!
- Compare with online, batch and mini-batch version?
- Keep separations between lambda, momentum and eta in the implementation (slide 66 NOTE ON NN - part 2). ✓
- Use high number of units but with regularization.
- Use sigmoid function for classification in output units with threshold (we may implement also softmax, cross entropy, etc.). ✓
- Use exhaustive grid-search to find best hyperparameters values (MODELS SELECTION).
Do it only for hyperparameters directly related to the VC-dim.
- Insert also STANDARD DEVIATION in the report, not only the MEAN of the error!! (K-FOLD CV with k=5 or 10...).
- Use CROSS-VALIDATION for training and validation set and then use a separate test set.
- Compute mean error for training, validation and test set considering different trials (initialization of the weights).
- Verify the model on different dataset (MONK + ML CUP).
- Describe in the report how we get the last model from the validation phase!
- Use MLP with backpropagation, momentum and L2 regularization. ✓
- Compare our simulator with an "oracle" tool to assess its correctness (Keras, Pytorch, etc.).
- Try to implement an efficient code for the experiments!
- Implement some decay technique for learning rate ✓
LEARNING RATE IN [0.01 0.5] in LMS WITHOUT MOMENTUM
LEARNING RATE IN [0.2, 0.9] in LMS WITH MOMENTUM
MOMENTUM IN [0.5, 0.9]
INIT_WEIGHTS IN [-0.01, 0.01], [-0.25, 0.25], [-0.1, 0.1]
MOMENTUM IN 

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
- Some trials for screening phase
- Model Selection
    - Hold-out ✓
    - K-fold cross validation ✓
    - Grid Search
    - Random Search
