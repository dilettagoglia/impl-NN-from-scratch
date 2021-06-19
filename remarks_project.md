## NEURAL NETWORK MODEL IMPLEMENTATION REMARKS
1. Use 1-hot encoding for categories.
2. Do we have to do preprocessing?
3. DON'T USE TEST SET!!! ONLY FOR MODEL EVALUATION!!
4. Use LMS (divide by mb for mini-batch) as error metric (we may implement also other metrics like MEE).
5. Initialize weights by random values near zero (or initialization from the paper by Bengio and Glorot).
6. Try a random starting number of trials/configurations (initial weights).
7. Use mini-batch version (aka SGD).
8. Check the learning curve for learning rate.
9. Use MOMENTUM (we may implement other type of momentum).
10. Stop training based on some criteria in every trial (NOT WITH A FIXED VALUES OF EPOCHS!!).
11. Use Tikhonov regularization (weights decay)!! (we may implement other kind of regularization).
12. Report only the error term in the report (plots or tables), not the entire loss!
13. Compare with online, batch and mini-batch version?
14. Keep separations between lambda, momentum and eta in the implementation (slide 66 NOTE ON NN - part 2).
15. Use high number of units but with regularization.
16. Use sigmoid function for classification in output units with threshold (we may implement also softmax, cross entropy, etc.).
17. Use exhaustive grid-search to find best hyperparameters values (MODELS SELECTION).
Do it only for hyperparameters directly related to the VC-dim.
18. Insert also STANDARD DEVIATION in the report, not only the MEAN of the error!! (K-FOLD CV with k=5 or 10...).
19. Use CROSS-VALIDATION for training and validation set and then use a separate test set.
20. Compute mean error for training, validation and test set considering different trials (initialization of the weights).
21. Verify the model on different dataset (MONK + ML CUP).
22. Describe in the report how we get the last model from the validation phase!
23. USe MLP with backpropagation, momentum and L2 regularization.
24. Compare our simulator with an "oracle" tool to assess its correctness (Keras, Pytorch, etc.).
25. Try to implement an efficient code for the experiments!

## PYTHON IMPLEMENTATION REMARKS
1. Use static methods for utility classes (when we don't care about the property of the object)
2. Use @property decorator to implement setter and getter methods
3.  
- _var -> underscore before the name variable is meant as a hint to another programmer that a variable or method starting with a single underscore is intended for internal use.
It should be considered an implementation detail and subject to change without notice.
- __var -> this is also called name mangling; the interpreter changes the name of the variable in a way that makes it harder to create collisions when the class is extended later.