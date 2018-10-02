#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:07:46 2018

@author: rajavardhan
"""


###https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/

import pandas as pd

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names)  

##Preprocessing#####

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])

##You can see that the values in the y series are categorical. 
#However, neural networks work better with numerical data. 
#Our next task is to convert these categorical values to numerical values.
#But first let's see how many unique values we have in our y series. 

y.Class.unique()  

#We have three unique classes 'Iris-setosa', 'Iris-versicolor' and 'Iris-virginica'.
# Let's convert these categorical values to numerical values.
# To do so we will use Scikit-Learn's LabelEncoder class.

from sklearn import preprocessing  
le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform) 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)



############# FEATURE SCALING ####################
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  



#############Training and Predictions############

from sklearn.neural_network import MLPClassifier  

#we will create three layers of 10 nodes each. 
#There is no standard formula for choosing the number of layers and nodes for a neural network and it varies quite a bit depending on the problem at hand.
#The best way is to try different combinations and see what works best.
mlp = MLPClassifier(hidden_layer_sizes=(10, 20, 30), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))

##Results can be slightly different from these because train_test_split randomly splits data into training and test sets,
#so our networks may not have been trained/tested on the same data.


























