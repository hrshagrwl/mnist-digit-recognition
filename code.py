#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test.py
# Author: hagrawa2 <hagrawa2@ncsu.edu>

import numpy as np
from mlxtend.data import loadlocal_mnist

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Function to store the predicted result into a CSV file as required
def createCSV(result, name):
    arr = np.zeros((len(result), 10), dtype = 'int32')
    arr = np.matrix(arr)
    
    for index, val in enumerate(result):
        arr[index, val] = 1
        
    np.savetxt(name, arr, fmt = '%i', delimiter = ",")

def load_data():
  files = {
    "X": "data/train-images-idx3-ubyte",
    "y": "data/train-labels-idx1-ubyte",
    "X_test": "data/t10k-images-idx3-ubyte",
    "y_test": "data/t10k-labels-idx1-ubyte"
  }
  X, y = loadlocal_mnist(images_path = files['X'], labels_path = files['y'])
  X_test, y_test = loadlocal_mnist(images_path = files['X_test'], labels_path = files['y_test'])
  return X, X_test, y, y_test 

def main():
  X, X_test, y, y_test = load_data()
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

  print('-----------------------------------lbfgs-------------------------------------------')
  # Logistics Regression
  lr = LogisticRegression(n_jobs=2, max_iter = 500, multi_class='multinomial', solver='lbfgs')
  lr.fit(X_train, y_train)

  y_train_pred_lr = lr.predict(X_train)
  acc_train_lr = accuracy_score(y_train_pred_lr, y_train)
  print('Accuracy on Training Dataset - Logistics Regression: {}'.format(acc_train_lr))

  y_val_pred_lr = lr.predict(X_val)
  acc_val_lr = accuracy_score(y_val_pred_lr, y_val)
  print('Accuracy on Validation Dataset Logistics Regression: {}'.format(acc_val_lr))

  print('-----------------------------------saga-------------------------------------------')

  lr = LogisticRegression(n_jobs=2, max_iter = 500, multi_class='multinomial', solver='saga')
  lr.fit(X_train, y_train)

  y_train_pred_lr = lr.predict(X_train)
  acc_train_lr = accuracy_score(y_train_pred_lr, y_train)
  print('Accuracy on Training Dataset Logistics Regression: {}'.format(acc_train_lr))

  y_val_pred_lr = lr.predict(X_val)
  acc_val_lr = accuracy_score(y_val_pred_lr, y_val)
  print('Accuracy on Validation Dataset Logistics Regression: {}'.format(acc_val_lr))

  print('-----------------------------------Final Model ----------------------------------------')

  lr = LogisticRegression(n_jobs=2, max_iter = 500, multi_class = 'multinomial', solver= 'lbfgs')
  lr.fit(X, y)

  y_pred_lr = lr.predict(X_test)
  acc_lr = accuracy_score(y_pred_lr, y_test)
  print('Accuracy on Testing Dataset Logistics Regression: {}'.format(acc_lr))

  createCSV(y_pred_lr, "lr.csv")

  print('-----------------------------------Random Forest----------------------------------------')

  # Random Forest Classifier
  rfc = RandomForestClassifier(n_jobs = 1, n_estimators = 500, max_features = 'auto', random_state = 42)
  rfc.fit(X_train, y_train)
  
  y_train_pred_rf = rfc.predict(X_train)
  acc_train_rf = accuracy_score(y_train_pred_rf, y_train)
  print('Accuracy on Training data Random Forest Classifier: {}'.format(acc_train_rf))

  y_val_pred_rf = rfc.predict(X_val)
  acc_val_rf = accuracy_score(y_val_pred_rf, y_val)
  print('Accuracy of Validation data Random Forest Classifier: {}'.format(acc_val_rf))

  print('-----------------------------------Final Random Forest----------------------------------------')

  rfc = RandomForestClassifier(n_jobs = 1, n_estimators = 500, max_features = 'auto', random_state = 42)
  rfc.fit(X, y)

  y_pred_rf = rfc.predict(X_test)
  acc_rf = accuracy_score(y_pred_rf, y_test)
  print('Accuracy on Testing data Random Forest Classifier: {}'.format(acc_rf))

  createCSV(y_pred_rf, "rf.csv")

if __name__ == "__main__":
  main()