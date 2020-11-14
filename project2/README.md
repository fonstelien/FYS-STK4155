## FYS-STK4155 Project 2 H20
You will find my report `project2.pdf` in the `report` folder and accompanying source code in the `source` folder. The `*.py` files are arranged such that the `utils.py` file contains all the common stuff for importing packages, creating matrices, evaluating MSE, R2 etc. In the other files, `ols.py`, `ridge.py`, `logistic.py`, and `neural.py` you will find the methods I have made for running each of the algorithms. Even if I usually don't really like to object-orient my code (it's amazing how seldom that actually is necessary...), for the implementation of Stochastic Gradient Descent and the neural netork, I couldn't find a way past it. Sometimes it's just the right thing to do. So, the SGD implementation is found in `StochasticGradientDescent.py`, where you will find an `SGD` class which is to be initialized with some parameters and sent in to the methods for OLS, Ridge and Logistic regression. See the docstrings. In `FeedForwardNeuralNetwork.py` you will find two classes; `Layer` which are the layers in the `FFNN` class further down. The `Layer`s are instantiated by the `FFNN` when you do an `FFNN.add_layer()`. Again, see the docstring.

I have used `jupyter lab` extensively in this work, during the code development process and for generating plots of my results. Scripting `plt` will never be beautiful, so please don't look too carefully. If you want to run some code for yourself, I have added a `conda` environment file here for you. You can install it by running
```
$ conda env create -f fys-stk4155.yml
```

... and you get rid of it again by
```
$ conda env remove --name fys-stk4155
```

Some test runs are also given in `testruns.ipynb` in the `source` folder. Enjoy! :whale:
