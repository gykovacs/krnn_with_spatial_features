#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:31:44 2018

@author: gykovacs
"""

# importing the dataset package from sklearn
import sklearn.datasets as sd

# import the KRNN_SF classifier
from KRNN_SF import KRNN_SF

# loading the IRIS dataset
X, y= sd.load_iris(return_X_y= True)

# turning the IRIS multi-classification problem into an unbalanced binary classification
y[y == 2]= 1

# fitting and predicting
krnn_sf= KRNN_SF(correction= 'r1')
krnn_sf.fit(X, y)
krnn_sf.predict_proba(X)