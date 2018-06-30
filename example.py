#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:31:44 2018

@author: gykovacs
"""

# importing the dataset package from sklearn
import sklearn.datasets as sd

from sklearn import metrics

# import the KRNN_SF classifier
from KRNN_SF import KRNN_SF

# loading the IRIS dataset
X, y= sd.load_iris(return_X_y= True)

# turning the IRIS multi-classification problem into an unbalanced binary classification
y[y > 1]= 1

# fitting and predicting
krnn_sf= KRNN_SF(correction= 'r2')
krnn_sf.fit(X, y)

y_pred = krnn_sf.predict_proba(X)

fpr, tpr, thresholds = metrics.roc_curve(y, y_pred[:,1])

print(metrics.auc(fpr, tpr))

