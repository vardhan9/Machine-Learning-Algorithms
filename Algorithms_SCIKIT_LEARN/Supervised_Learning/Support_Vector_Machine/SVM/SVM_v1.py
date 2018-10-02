#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:10:40 2018

@author: rajavardhan
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# Importing data with url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

names = ['variance','skewness','curtosis','entropy','class']

dataset = pd.read_csv(url, names=names)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,4]


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print ("Accuracy of this model is ",int(accuracy_score(y_test,y_pred)*100),"percent") 