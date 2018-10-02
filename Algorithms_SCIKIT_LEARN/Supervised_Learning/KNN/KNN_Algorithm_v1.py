#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 11:29:44 2018

@author: rajavardhan
"""
# this KNN ALGORITHM . 
# Reference : https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

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

# Feature Scaling

# FOR CLARIFICATION read https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)


X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=7)  
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score   
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
print ("Accuracy of this model is ",int(accuracy_score(y_test,y_pred)*100),"percent")
