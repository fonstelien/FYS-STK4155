## FYS-STK4155 Project 3 H20
You will find my report `project3.pdf` in the `report` folder and accompanying source code in the `source` folder. The `*.py` files are arranged such that the `utils.py` file contains all the common stuff like input file parsing and evaluation. In the other files, `autoregression.py` and `neural_network.py` you will find the methods I have made for preparing the data, reshaping input for `keras`, model evaluation, etc. See the docstrings for usage. The dataset which I have used in this report is found under `source/data`. 

This report is about the creation of a so-called digital twin for the cooling system of a power transformer using machine learning methods. Available for creating this model, I have a dataset containing operational data from a transfomer in an industrial application. My background is in electrical engineering, and I suppose that will shine through when you read the report. I have tried not to be too exhaustive with the details about transformers, but a review of what I believe the reader would need to know is given. 

I have based part of my kode on `keras` for creating an RNN model, and I did not think it was necessary to wrap that in a function in a `.py` file, but run it directly in `jupyter lab` instead. This project has a lot less code than did the former two, but a lot of effort went into preparing and understanding the dataset -- I have focused more on the "Applied data analysis" part of the course's title this time.

I have used `jupyter lab` extensively in this work, during the code development process and for generating plots of my results. Scripting `plt` will never be beautiful, so please don't look too carefully. If you want to run some code for yourself, I have added a `conda` environment file here for you. You can install it by running
```
$ conda env create -f fys-stk4155.yml
```

... and you get rid of it again by
```
$ conda env remove --name fys-stk4155
```

Enjoy! :whale:
