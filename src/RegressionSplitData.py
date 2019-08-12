# ''' Wed Apr 24 09:09:08 2019; By Rojan Shrestha '''

import sys,os
import argparse, logging
from time import time

import pandas as pd
import numpy as np

import pickle
import datetime 

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Features import DataProcess

class TimeSeriesData (DataProcess):

  def __init__(self, fo_train_ratio, fo_test_ratio, 
               in_back_offset=2016, in_forward_offset=72, in_step=6):
    """
    in_back_offset: offset backward 
    in_forward_offset: offset forward 
    in_step: stepsize   
    """

    super().__init__(fo_train_ratio, 0.0, fo_test_ratio)
    self._in_back_offset    = in_back_offset 
    self._in_forward_offset = in_forward_offset 
    self._in_step           = in_step 

  def split_dataset(self, standardize=False):
    """
    Aims: - split dataset into three groups - training, validation, and testing
            based on their assigned proportion. Training and validation were considered as 
            training.

    Params: standardize=True will use standardscalar to normalize 
    """
    try:
      in_total_sample = self._ar_values.shape[0]
      self._ts_indexes = [] # keep index of test data

      #1. training
      in_ubound = int(self._fo_train_ratio * in_total_sample)
      print("INFO: Training data")
      print("===================")
      print("INFO: training data indexing from 0-{}".format(in_ubound))
      print("INFO: # of training samples {}".format(in_ubound))
      ar_trainX, ar_trainY = self.make_dataset(0, in_ubound, True, keep_index=False)

      self._ar_train_X = ar_trainX 
      self._ar_train_Y = ar_trainY
      print("INFO: final selection for training: {} samples with step size {}".format(self._ar_train_X.shape[0], self._in_step))
      print("...")

      #2. testing 
      print("INFO: Test data")
      print("===============")
      in_lbound = in_ubound + 1 
      in_ubound = in_lbound + int(self._fo_test_ratio * in_total_sample)
      print("INFO: testing data indexing from {}-{}".format(in_lbound, in_ubound))
      print("INFO: # of testing samples {}".format(int(self._fo_test_ratio * in_total_sample)))

      ar_testX, ar_testY = self.make_dataset(in_lbound, None, False, keep_index=True)
      self._ar_test_X = ar_testX 
      self._ar_test_Y = ar_testY
      print("INFO: final selection for testing: {} samples with step size {}".format(self._ar_test_X.shape[0], self._in_step))
      print("...")

      if standardize:
        self._oj_std_scale = StandardScaler()
        self._oj_std_scale.fit(ar_trainX)
        self._ar_train_X = self._oj_std_scale.transform(ar_trainX)
        self._ar_test_X = self._oj_std_scale.transform(ar_testX)

    except IndexError: 
      print("EXCEPT: no element in array")
      sys.exit(0)


  def make_dataset(self, in_min_idx=0, in_max_idx=120000, random=False, keep_index=False):
    ''' 
        Aims: Create target variable and input features. 
              Moving average was computed with step size to avoid redundancy.  

        Params: 
            in_min_idx: lower index 
            in_max_idx: upper index 
            random: shuffle 
            keep_index: to keep index of original array, if true is inputted
    ''' 
    if in_max_idx is None: in_max_idx = self._ar_values.shape[0] - 1 # - self._in_forward_offset - 1
    in_idx = in_min_idx + self._in_back_offset

    in_rows     = (in_max_idx - self._in_forward_offset - in_idx) // self._in_step
    # in_rows     = (in_max_idx - self._in_forward_offset - in_idx)
    in_axis2    = self._ts_num_idxes.shape[0] + self._ts_str_idxes.shape[0]
    ar_samples  = np.zeros((in_rows, in_axis2))
    ar_targets  = np.zeros((in_rows, 1))
    if random:
      print("INFO: random selection")
      ar_rnd_idx = np.random.randint(in_idx, in_max_idx, in_rows)
      for i in range(ar_rnd_idx.shape[0]):
        in_idx          = ar_rnd_idx[i] 
        # indices         = np.arange(in_idx-self._in_back_offset, ar_rnd_idx[i], self._in_step)
        indices         = np.arange(in_idx-self._in_back_offset, ar_rnd_idx[i], 1)
        ar_samples[i, self._ts_num_idxes]   = np.mean(self._ar_values[indices[:, None], self._ts_num_idxes], axis=0) 
        if self._ts_str_idxes.shape[0] > 0: 
          ar_samples[i, self._ts_str_idxes]   = np.median(self._ar_values[indices[:, None], self._ts_str_idxes], axis=0).astype(int)
        ar_targets[i]   = self._ar_values[in_idx + self._in_forward_offset, self._in_target_idx]
        if keep_index: self._ts_indexes.append(in_idx)
    else:
      print("INFO: sequential selection")
      # after preprocessing the data, dimension of new array 
      in_j = 0
      while in_j < ar_samples.shape[0]:
        indices          = np.arange(in_idx-self._in_back_offset, in_idx, 1)
        ar_samples[in_j, self._ts_num_idxes] = np.mean(self._ar_values[indices[:, None], self._ts_num_idxes], axis=0)
        if self._ts_str_idxes.shape[0] > 0: 
          ar_samples[in_j, self._ts_str_idxes]   = np.median(self._ar_values[indices[:, None], self._ts_str_idxes], axis=0).astype(int)
        ar_targets[in_j] = self._ar_values[in_idx + self._in_forward_offset, self._in_target_idx]
        if keep_index: self._ts_indexes.append(in_idx)
        in_idx          += self._in_step
        in_j            += 1
    return ar_samples, ar_targets.ravel()

