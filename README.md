# impl-NN-from-scratch
Implementation of a Neural Network from scratch in Python for the Machine Learning course.

<img src="https://elearning.agr.unipi.it/pluginfile.php/4/course/section/13/marchio_unipi_pant541.png" alt="unipi" width="40%" />

<br/>

#### Authors:

- [Diletta Goglia](https://github.com/dilettagoglia) - M.Sc.  in Artificial Intelligence, University of Pisa

- [Paolo Murgia](https://github.com/Musca23) - M.Sc.  in Artificial Intelligence, University of Pisa

## Description
Project implementation for Machine Learning exam, Master's Degree Course in Computer Science, Artificial Intelligence curriculum, University of Pisa.

Professor: [Alessio Micheli](http://pages.di.unipi.it/micheli/).

For more further info please read the [report](GOGLIA_MURGIA_Report.pdf).

### Abstract
The project consists in the implementation of an Artificial Neural Network built from scratch using Python, without using pre-built libraries. 
The overall validation schema consists in a preliminary screening phase to reduce the hyperparameters search space, followed by a first coarse grid-search and a second but finer one. All the explored models are validated with a 5-fold cross validation.
The resulting model is a 2 hidden layer network with 20 units each and ReLU activation for both layers. 

### Code implementation.
For clarity, transparency and accessibility purposes, we decided to write our code 
following the ”tacit and explicit conventions applied in Scikit-learn and its API”, 
and soto follow the notation of the [glossary](https://scikit-learn.org/stable/glossary.html#glossary-parameters),
 eg.  using standard terms for methods, attributes, etc.  
 This well-known ”best practice” allowed us to write a good-quality code, well-commented 
 and easy for reading, understanding and experiments reproducibility.


## References

Useful sources used & documentation:
- [Property vs. Getters and Setters in Python - *Datacamp*](https://www.datacamp.com/community/tutorials/property-getters-setters?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=229765585183&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1008645&gclid=Cj0KCQjwlMaGBhD3ARIsAPvWd6hxk3HTgP9NpO_kbD2pgOt2N0bDLH2zivo6B_y0O7xHkyT5FITRFI4aArXHEALw_wcB)
- [Numpy documentation](https://numpy.org/doc/stable/)
- [Implementing a Neural Network from Scratch in Python - *WildML*](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [tqdm (progress bar)](https://pypi.org/project/tqdm/)
- [Building a Neural Network From Scratch Using Python (Part 1)](https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432)
- [How to build your own Neural Network from scratch in Python](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
- [How to code a neural network from scratch in Python](https://anderfernandez.com/en/blog/how-to-code-neural-network-from-scratch-in-python/)
- [Naming with Underscores in Python](https://medium.com/python-features/naming-conventions-with-underscores-in-python-791251ac7097)
- [Hyperparameters tuning for ML](https://towardsdatascience.com/how-to-tune-hyperparameters-for-machine-learning-aa23c25a662f)
- [Early stopping for training](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)
- [Glorot initialization for weights](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
- [Performance comparison for Monk's problems](https://www.researchgate.net/publication/2293492_The_MONK's_Problems_A_Performance_Comparison_of_Different_Learning_Algorithms/link/57358d6208ae9f741b2987fb/download)
- [Grid Search in Python from scratch](https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b)
- [itertools — Functions creating iterators for efficient looping¶](https://docs.python.org/3/library/itertools.html)
- [Joblib: running Python functions as pipeline jobs](https://joblib.readthedocs.io/en/latest/)
- [The MONK's Problems A Performance Comparison](https://www.researchgate.net/publication/2293492_The_MONK%27s_Problems_A_Performance_Comparison_of_Different_Learning_Algorithms)

For parameter tuning:
- [How to Tune Hyperparameters for Machine Learning](https://towardsdatascience.com/how-to-tune-hyperparameters-for-machine-learning-aa23c25a662f)
- [Hyper-parameter selection and tuning](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-i-hyper-parameter-8129009f131b)
- [What are Hyperparameters? and How to tune the Hyperparameters](https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a)
- [L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
- [A beginner guide to bias and variance in ML](https://medium.com/analytics-vidhya/a-beginner-guide-to-bias-and-variance-in-ml-c016fbb502ea)

Comparison with pre-built models:
- [Scikit MLP Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
- [Keras Classifier](https://www.tensorflow.org/api_docs/python/tf/keras/wrappers/scikit_learn/KerasClassifier)

