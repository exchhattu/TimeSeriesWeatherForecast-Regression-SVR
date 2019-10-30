# Time series data - weather forecast
  
## **Motivation**
* Predict a temperature of the next few hours using multiple machine learning
 algorithm.

## **Requirements** 
* Python (3.6.0)
* Pandas (0.24.1)
* NumPy (1.16.0)
* Keras (2.2.4)
* TensorFlow (1.13.1)
* Juypter (4.4.0)
* Matplotlib (3.0.2) and Seaborn (0.9.0)

## **Dataset Overview**
Dataset used for **weather forecasting** was downloaded from the 
book [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks).
The dataset contains recorded weather data comprising of 13 different features from the year 2009 to 2016.
The record was in 10 minutes difference, which yields 6 samples per hour and 52557 samples per year. 
For 8 years, the number of samples is 420550. This dataset is dividing into training and testing groups in the ratio of 8:2.

## **Method**
* Linear regression (LR) with different regularization and loss function was
  tried.
* Support vector regression (SVR) was also tested.
* For cross-validation, [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) were tested. 
However, RandomizedSearchCV was selected afterward because it produced the superior performance 
in a limited computing resource. 

## **Programs**
* A program was developed using linear regression (regularization methods) and support vector regression. 

### **Usage:**
```python
$ python3 ./LRModelRandCV.py -h 
```

### **Example**
* Linear regression
```
$ python3 ./LRModelGridCV.py -t -n 8 -i ../data/jena_climate_2015_2016.csv 
  
  -t: includes date and time column feature otherwise it is excluded 
  -n: # of CPUs used to train a model.
```

* Selection of regularization algorithm 
```python
$ python3 ./src/LRModelRandCV.py -t -i ./data/jena_climate_2009_2016.csv -n 8 -m lr_ri
 -m accepts lr_ri for ridge, lr_la for lasso, lr_enet for elastic net, and lr_sgd for stochastic gradient regressor
```

* Support vector regressor
```python
$ python36 ./src/SVR.py -t -i ./data/jena_climate_2009_2016.csv -n 8
```

## **Performance**
* Linear Regression showed the best performance of [MSE 3.10](https://github.com/exchhattu/TimeSeriesWeatherForcast/blob/master/Notebook-Analysis/Weather.ipynb).

* Various flavors (lasso, ridge, elastic net and stochastic gradient descent) of linear regression were tested
but the performance was not improved over ordinary linear regression. Lasso showed the worst performance with MSE 3.53;
however, the remaining versions showed either similar or worse than ordinary linear regression.
The randomized algorithm was used in cross-validation to find the best solution where different value of the alpha parameter
of Ridge or Lasso was searched. The best performance was obtained with alpha is 0, which indicates that the regularization term
does not contribute to achieve the better performance.

* Multiple parameters such as alpha and l<sub>1</sub> ratio were explored for Elastic net and SGD using same searching strategy,
the best result for these methods are 3.27 (elastic net with alpha=0.58 and l<sub>1</sub> ratio=0.03)
and 3.19 (SGD with alpha=0.58, l<sub>1</sub> penalty and l<sub>1</sub> ratio=0.51.
These results could not beat the ordinary linear regression performance.

* [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) took multiple weeks for training and validation however it showed the worst performance of with 14.31 MSE.
