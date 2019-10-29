# Time series data - weather forecast
  
## **Motivation**
* Predict a temperature of the next few hours using multiple machine learning
 algorithm.

## **Background**
* Dataset used for **weather forecasting** was downloaded from the book [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks).
* Dataset contains mostly numerical data and only one categorical data.

## **Requirements** 
* Python (3.6.0)
* Pandas (0.24.1)
* numpy (1.16.0)
* keras (2.2.4)
* Tensorflow (1.13.1)
* Juypter (4.4.0)
* Matplotlib (3.0.2) and Seaborn (0.9.0)

## **Dataset Overview**
The dataset contains recorded weather data comprising 13 different features from 2009 to 2016.
Most columns contain numeric value except Date and Time column, other c. From Date and Time attribute,
few extra attributes were generated.

Each record is in 10 minutes difference, which yields 6 samples per hour and 52557 samples per year. For 8 years, the number of samples is 420550.
This dataset is dividing into training and testing groups in the ratio of 8:2.

## **Method**
* Linear regression (LR) with different regularization and loss function was used.
* Support vector regression (SVR) was also tested.
* Recurrent neural network with different options were also tested.
* Cross-validation methods such as [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) were tested, but RandomizedSearchCV was used due to the superior performance in limited computing resource. 

## **Programs**
* Program, which wraps multiple machine learning algorithms, was developed. The encapsulated programs are linear regression with different regularization methods, support vector regression, and RNN using Keras (added later). Here is the way how this program can be used.
* Although multiple approaches were created, random search cross-validation was used in final benchmarking since it showed the better result with a limited computing resource.

### **Usage:**
```python
$ python3 ./LRModelRandCV.py -h 
```

### **Example**
* Linear regression
```
$ python3 ./LRModelGridCV.py -t -n 8 -i ../data/jena_climate_2015_2016.csv 
  
  -t: includes date and time column otherwise it will use other columns with numeric values 
  -n: # of CPUs used to train a model.
```

* Selection of regularization 
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

* Various flavors (lasso, ridge, elastic net and stochastic gradient
descent) of linear regression were tested but it could not improve the performance over linear regression.
Lasso showed the worst performance of MSE 3.53 whereas ridge performance was similar to linear regression.
This was obtained while searching the optimal value of alpha, which is found to be 0, using a randomized cross-validation search.

* Although multiple parameters such as alpha, l<sub>1</sub> ratio were explored for Elastic net and SGD using same searching strategy, the best result for these methods are 3.27 (elastic net with alpha=0.58 and l<sub>1</sub> ratio=0.03) and 3.19 (SGD with alpha=0.58, l<sub>1</sub> penalty and l<sub>1</sub> ratio=0.51, which did not beat the linear regression performance.

* [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) took multiple weeks for training and validation however it showed the worst performance of with 14.31 MSE.
