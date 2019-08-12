# ''' Wed Apr 24 09:09:08 2019; By Rojan Shrestha '''

import sys,os
import argparse, logging
from time import time

import pandas as pd
import numpy as np


import pickle
import datetime 

from scipy.stats import uniform

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import TimeSeriesSplit 
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression, mutual_info_regression

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import randint  


from Features import DataProcess
from RegressionSplitData import TimeSeriesData

class SVMregression(TimeSeriesData):

  def __init__(self, fo_train_ratio, fo_test_ratio, in_back_offset=2016, in_forward_offset=72, in_step=6, train_test_split=5):
    self._train_test_split = train_test_split 
    super().__init__(fo_train_ratio, fo_test_ratio, in_back_offset, in_forward_offset, in_step) 

  def save_trained_model(self):
    r'''
      Aims: to save a trained model.
    '''
    # save grid search model 
    ts_time_details = datetime.datetime.now().strftime("%d %b %y %H %M %S").split()
    st_fname = "TM_SVR_%s.gs" %"_".join(ts_time_details)
    with open(st_fname, 'wb') as oj_file:
      pickle.dump(self._gs_grid, oj_file)
    print("INFO: dumped {}".format(st_fname))

  def print_model_param(self, n_top=5):
    r'''
      Aims: to print the training and validating models performance  
      params:
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
        print("Parameters: {0}".format(oj_results['params'][candidate]))
        print("---")

  def estimate(self, mode="", params="known", randomize=False, njob=1):
    """
    Aims: - model parameters using given data
          - if saved model existed, read from given file

    params:
        mode:   "prebuild" or None. If prebuilt is used, path should be provided.  
        params: "known" or "determine". known is parameter obtained from training on small set
                and further used to train on large set to avoid many combination in grid or
                randomize cross-validation (CV).
        randomize: True for RandomizeCV and False for gridSearchCV 
        njob: # of CPUs
    """

    if mode == "prebuilt":
      self._gs_grid = pickle.load(prebuild_model)
    else:
      if params == "known":
        print("INFO: using known parameter")
        pe_data_trans   = ColumnTransformer(transformers=[('in', StandardScaler(), self._ts_num_idxes)])
        if self._ts_str_idxes.shape[0]>0:
          pe_data_trans   = ColumnTransformer(transformers=[('in', StandardScaler(), self._ts_num_idxes),
                                                          ('st', OneHotEncoder(), self._ts_str_idxes)])
        self._gs_grid     = Pipeline(steps = [('oj_trans', pe_data_trans), 
                                              ("oj_svm_clf", SVR(kernel="rbf", C=9315.84, epsilon=0.04, gamma=0.06))])
      else:
        pi_rbf_kernel_svm = Pipeline([ ("oj_svm_clf", SVR(kernel="rbf"))])
        if self._ts_str_idxes.shape[0]>0:
          pe_data_trans = ColumnTransformer(transformers=[('in', StandardScaler(), self._ts_num_idxes),
                                                          ('st', OneHotEncoder(), self._ts_str_idxes)])
        else:
          pe_data_trans = ColumnTransformer(transformers=[('in', StandardScaler(), self._ts_num_idxes)])
        pi_rbf_kernel_svm = Pipeline(steps = [('oj_trans', pe_data_trans), 
                                              ("oj_svm_clf", SVR(kernel="rbf"))])
        ts_fo_gammas    = list([1e-10, 1e-5, 1e-1]) 
        ts_fo_Cs        = list([100, 1000, 10000]) # regularization 
        ts_fo_eps       = list([1e-10 ,1e-5, 1e-1]) # regularization 
        param_grid      = [{'oj_svm_clf__C': ts_fo_Cs,
                            'oj_svm_clf__epsilon': ts_fo_eps,
                            'oj_svm_clf__gamma': ts_fo_gammas }]
        ti_cv         = TimeSeriesSplit(n_splits= self._train_test_split)
        self._gs_grid = GridSearchCV(pi_rbf_kernel_svm, cv=ti_cv, param_grid=param_grid, n_jobs=njob)
        if randomize: 
          print("INFO: randomize...")
          ts_st_kernel    = ['linear', 'poly', 'rbf', 'sigmoid']
          ts_fo_gammas    = uniform(1e-10, 1e-1) 
          ts_fo_Cs        = uniform(1, 10000) # regularization 
          ts_fo_eps       = uniform(1e-10, 1e-1) # regularization 
          param_grid      = {'oj_svm_clf__kernel': ts_st_kernel,
                             'oj_svm_clf__C': ts_fo_Cs,
                             'oj_svm_clf__epsilon': ts_fo_eps,
                             'oj_svm_clf__gamma': ts_fo_gammas }
          self._gs_grid = RandomizedSearchCV(pi_rbf_kernel_svm, 
                                             cv=ti_cv, param_distributions=param_grid, 
                                             n_iter=30, n_jobs=njob)

      print("INFO: training starts ...")
      t0 = time()
      self._gs_grid.fit(self._ar_train_X, self._ar_train_Y)
      print("INFO: finished train in %0.3fs" % (time() - t0))
      self.save_trained_model()
      if not params == "known": self.print_model_param(n_top=3)

  def guess(self):
    print("INFO: prediction using fine tuning parameters...")
    ar_pred_Y   = self._gs_grid.predict(self._ar_test_X)
    fo_mse      = mean_squared_error(self._ar_test_Y, ar_pred_Y)
    fo_rmse     = np.sqrt(fo_mse)
    print("INFO: RMSE %3.2f " %fo_rmse)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='SVR.py', description='Support vector regression model')
  parser.add_argument('-i','--path', required=True, help='input path for data')
  parser.add_argument('-m','--mode', required=False, default="", help='" " or prebuilt')
  parser.add_argument('-p','--param', required=False, default="", help='"known" or " "')
  parser.add_argument('-n','--njob', default=1, required=False, type=int)
  parser.add_argument('-c','--randcv', required=False, default=True, action='store_true')
  parser.add_argument('-e','--exclude_idx', default="", required=False, type = str, help="1,3,4 or 1-2,5 or 1-4")
  parser.add_argument('-l','--exclude_col', default="", required=False, type = str, help="colname2,colname2")
  parser.add_argument('-t','--use_date', required=False, default=False, action='store_true')
  oj_args = parser.parse_args(sys.argv[1:])

  oj_ti_data = SVMregression(0.80, 0.20, 2016, 72, 6, 10)
  oj_ti_data.read_data(oj_args.path, use_date=oj_args.use_date, exclude_column_idxes=oj_args.exclude_idx, exclude_columns=oj_args.exclude_col)
  # oj_ti_data.read_data(oj_args.path, True)
  oj_ti_data.identify_outlier(5.0)
  oj_ti_data.split_dataset()
  oj_ti_data.estimate(mode=oj_args.mode, params=oj_args.param, randomize=oj_args.randcv, njob=oj_args.njob)
  oj_ti_data.guess()

