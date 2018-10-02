#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:24:44 2018

@author: rajavardhan
"""

# Reference : https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

# importing Libraries
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  

# Importing data with url
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(url, names=names) 

# Slicing the columns

X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values

# Splitting the data into Test and Train

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

## POLYNOMIAL KERNEL

from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly',degree=12)  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print ("Accuracy with POLYNOMIAL Kernel is ",int(accuracy_score(y_test,y_pred)*100),"percent") 

## GAUSSIAN KERNEL

from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print ("Accuracy with GAUSSIAN Kernel is ",int(accuracy_score(y_test,y_pred)*100),"percent") 

## SIGMOID KERNEL

from sklearn.svm import SVC  
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print ("Accuracy with SIGMOID Kernel is ",int(accuracy_score(y_test,y_pred)*100),"percent")

## NOTE : sigmoid kernel performs the worst. 
#This is due to the reason that sigmoid function returns two values, 0 and 1, 
#therefore it is more suitable for binary classification problems.
