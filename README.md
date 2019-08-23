# Time series data - weather forecast

## **Motivation**
* Predict temperature of the next few hours using multiple machine learning
  algorith. 

## **Background**
* Dataset used for **weather forcasting** was from book [Deep Learning with Python](https://github.com/fchollet/deep-learning-with-python-notebooks). 
* Dateset contains mostly numerical data and only one categorical data. 
* It was used in the book to teach reccurent neural network (RNN).

## **Requirements** 
* Python (3.6.0)
* Pandas ()
* numpy (1.16.0)
* keras (2.2.4) 
* Tensorflow (1.13.1)
* Juypter ()
* Matplotlib and Seaborn

## **Dataset Overview**
Dataset contains recorded weather data comprising 13 different features from 2009 to 2016. 
Except Date and Time column, other columns contain numeric data. From Date and Time attribute, 
few extra attributes were genereated. 

Each record is in 10 minutes difference, which yield 6 samples per hour and 52557 
samples per year. For 8 years, the number of samples reach to 420550.
Generally, this dataset is dividing into training and testing groups in the
ratio of 8:2.  

## **Method**
* Multiple algorithms such as linear regression (LR) with different regulaization
  and loss function were used.
* Support vector regression (SVR) was also tested. 
* Recurrent neural network with different options were also tested.

* Cross-validation methods such as [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) and [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) were implemented, but RandomizedSearchCV was used due to superior performace in limited computing resource. These searching 
method is only used in LR and SVR. 

## **Programs**
* Program, which wraps multiple machine learning algorithms, was developed. The
encapsulated programs are linear regresion with different regularization
methods, supprot vector regression, and RNN using keras (added later). Here are the way how
these program can be used. 
* Although multiple approaches were created, random search cross validation was
used since it provided better result with limited computing resource. 

### **Usage:**
```python
$ python3 ./LRModelRandCV.py -h 
```

### **Example**
* Linear regression
```
$ python3 ./LRModelGridCV.py -t -n 8 -i ../data/jena_climate_2015_2016.csv 
  
  -t: includes date and time column otherwise it will use other numeric features
  -n: # of cpus used for training.
```

* Selection of regularization 
```python
$ python3 ./src/LRModelRandCV.py -t -i ./data/jena_climate_2009_2016.csv -n 8 -m lr_ri
 -m can accept lr_ri for ridge, lr_la for lasso, lr_enet for elastic net, and lr_sgd for stochastic gradient regressor
```

* Support vector regressor
```python
$ python36 ./src/SVR.py -t -i ./data/jena_climate_2009_2016.csv -n 8
```

## **Performance** 
* Linear Regression showed the best performance of 
[MSE 3.10](https://github.com/exchhattu/TimeSeriesWeatherForcast/blob/master/Notebook-Analysis/Weather.ipynb). 

* Different flavors for linear regression such as lasso, ridge, elastic net and stochastic gradient 
descent (SGD) were tested but it could not improve the performance over linear regression. 
Lasso showed worst performance of 3.53 whereas ridge performance was similar with linear regression. 
This was obtained while searching optimal value of alpha, which is found to be 0, using randomized 
cross validation search.  

* Although multiple parameters such as alpha, l<sub>1</sub> ratio were explored for Elastic net and SGD using 
same searching strategy, the best result for these methods are 3.27 (elastic net with alpha=0.58 and l<sub>1</sub> ratio=0.03) 
and 3.19 (SGD with alpha=0.58, l<sub>1</sub> penalty and l<sub>1</sub> ratio=0.51, which did not beat the linear regression
performance.

* [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) took multiple weeks for 
training and validation however it showed the worst performance of around 14.31 MSE. 
