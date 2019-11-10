''' Wed Apr 24 09:09:08 2019; By Rojan Shrestha '''

import sys,os
import argparse, logging
from time import time

import pandas as pd
import numpy as np


import pickle
import datetime 

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline

from scipy.stats import randint as randint
from scipy.stats import uniform

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression, mutual_info_regression

from sklearn.base import BaseEstimator, TransformerMixin


from Features import DataProcess
from RegressionSplitData import TimeSeriesData

class LReg(TimeSeriesData):

  def __init__(self, fo_train_ratio, fo_test_ratio, in_back_offset=2016, in_forward_offset=72, in_step=6, train_test_split=5):
    """
      Aims: constructor and wraps from super class.
      Params:
        fo_train_ratio:         % in the training set
        fo_test_ratio:          % in the testing set
        in_back_offset=2016:    # of rows taken backward from current row for prediction 
        in_forward_offset=72:   target variable  
        in_step=6:              step size used to consider a row selection 
        train_test_split=5:     used in time series data split for cross validation
    """
    self._train_test_split = train_test_split 
    super().__init__(fo_train_ratio, fo_test_ratio, in_back_offset, in_forward_offset, in_step) 

  def save_trained_model(self):
    r''' Aims: to save a trained model  '''
    # save grid search model 
    ts_time_details = datetime.datetime.now().strftime("%d %b %y %H %M %S").split()
    st_fname = "TM_lr_%s.gs" %"_".join(ts_time_details)
    with open(st_fname, 'wb') as oj_file:
      pickle.dump(self._gs_grid, oj_file)
    print("INFO: dumped {}".format(st_fname))

  def print_model_param(self, n_top=5):
    '''
      Aims: display training and validating models performance  
      Params:
        * n_top: top solutions for printing.
    '''
    oj_results = self._gs_grid.cv_results_

    for i in range(1, n_top + 1):
      # gives an index of non-zeros
      ar_best_candidates = np.flatnonzero(oj_results['rank_test_score'] == i)
      for candidate in ar_best_candidates:
        print("Rank: {0} model".format(i))
        print("Validation score: mean {0:.3f}, std: {1:.3f}".format(
               oj_results['mean_test_score'][candidate],
               oj_results['std_test_score'][candidate]))
        print("Parameters: {0}".format(oj_results['Params'][candidate]))
        print("---")

  def estimate(self, prebuilt_model=None, model="lr", njob=1):
    """
      Aims: prepare a pipeline required for model building 

      Params:
        prebuilt_model: path to prebuild model. None(default):    
        model:          pass the enum/label to select a particular model. lr(default) 
        njob:           # of cpus assigned. 1(default)
    """
    if prebuilt_model:
      print("INFO: reading prebuilt model for testing...")
      with open(prebuilt_model, 'rb') as oj_fhandle:
        self._gs_grid = pickle.load(oj_fhandle)
      return 

    # by default model is lr
    in_iter = 100 
    pi_lr = Pipeline(steps = [("oj_lr", LinearRegression())])
    param_grid = {'oj_lr__fit_intercept': [True, False]}
    if model == "lr_la":
      print("INFO: selected model - linear regression lasso")
      pi_lr         = Pipeline(steps = [("oj_lr_la", Lasso())])
      param_grid    = {'oj_lr_la__alpha': randint(0, 100)}
      in_iter       = 100
    elif model == "lr_ri":
      print("INFO: selected model - linear regression ridge")
      pi_lr = Pipeline(steps = [("oj_lr_ri", Ridge(fit_intercept=True, copy_X=True))])
      param_grid = {'oj_lr_ri__alpha': randint(0, 100)}
      in_iter   = 100
    elif model == "lr_enet":
      print("INFO: selected model - linear regression elasticnet")
      in_iter       = 100 
      pi_lr         = Pipeline(steps = [("oj_lr_re", ElasticNet(max_iter=10000))])
      param_grid    = {'oj_lr_re__alpha': uniform(0.05, 100), 'oj_lr_re__l1_ratio': uniform(0.0, 1.0)}
    elif model == "lr_sgd":
      in_iter       = 100 
      print("INFO: selected model - linear regression sgd")
      pi_lr = Pipeline(steps = [("oj_lr_sgd", SGDRegressor(fit_intercept=True, max_iter=10000))])
      param_grid = {'oj_lr_sgd__penalty': ['l2', 'l1', 'elasticnet'],
                    'oj_lr_sgd__alpha': uniform(0, 100),
                    'oj_lr_sgd__l1_ratio': uniform(0.0, 1.0),
                    'oj_lr_sgd__learning_rate' : ['optimal', 'constant', 'invscaling', 'adaptive']}


    ti_cv   = TimeSeriesSplit(n_splits= self._train_test_split)
    self._gs_grid = RandomizedSearchCV(pi_lr, cv=ti_cv, param_distributions=param_grid, n_iter=in_iter, n_jobs=njob)
    t0 = time()
    print("INFO: start train ...")
    self._gs_grid.fit(self._ar_train_X, self._ar_train_Y)
    print("INFO: finished train in %0.3fs" % (time() - t0))
    self.save_trained_model()
    self.print_model_param(n_top=3)

  def guess(self, write_result=False):
    """ 
      Aims: predict on test set using model parameters 
      Params: write_result: write date, original and predicted values 
    """

    print("INFO: saving model with fine tuning parameters...")
    ar_pred_Y   = self._gs_grid.predict(self._ar_test_X)
    fo_mse      = mean_squared_error(self._ar_test_Y, ar_pred_Y)
    fo_rmse     = np.sqrt(fo_mse)
    print("INFO: %3.2f " %fo_rmse)

    # writing the predicted label
    if write_result:
      ar_date = np.array(self._df_data.values[self._ts_indexes, 0])
      ar_Y_test_pred = np.column_stack((ar_date, self._ar_test_Y, ar_pred_Y))
      np.savetxt("test_result.out", ar_Y_test_pred, fmt="%s,%0.2f,%0.2f", delimiter=",")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='LRModelRandCV.py', description='Weather prediction using randomized cross validation')
  parser.add_argument('-i','--path', required=True, help='input path for data')
  parser.add_argument('-m','--method', default="lr", required=False, help='[lr, lr_la, lr_ri, lr_enet, lr_sgd]')
  parser.add_argument('-k','--known_model', default=None, required=False, help='Path to prebuilt model')
  parser.add_argument('-n','--njob', default=1, required=False, type = int, help="# of CPUs")
  parser.add_argument('-e','--exclude_idx', default="", required=False, type = str, help="1,3,4 or 1-2,5 or 1-4")
  parser.add_argument('-l','--exclude_col', default="", required=False, type = str, help="colname2,colname2")
  parser.add_argument('-t','--use_date', required=False, default=False, action='store_true')
  oj_args = parser.parse_args(sys.argv[1:])

  oj_ti_data = LReg(0.80, 0.20, 100, 72, 6, 10)
  oj_ti_data.read_data(oj_args.path, use_date=oj_args.use_date, exclude_column_idxes=oj_args.exclude_idx, exclude_columns=oj_args.exclude_col)
  oj_ti_data.identify_outlier(5.0)
  oj_ti_data.split_dataset(standardize=True)
  oj_ti_data.estimate(prebuilt_model=oj_args.known_model, model=oj_args.method, njob=oj_args.njob)
  oj_ti_data.guess(write_result=True)
