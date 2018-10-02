#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:13:20 2018

@author: rajavardhan
"""

## DATA : https://archive.ics.uci.edu/ml/datasets/Census+Income####

import pandas as pd
import numpy as np
import seaborn as sn

data = pd.read_csv("adult_data.txt")

A=pd.get_dummies(data["workclass"])
B=pd.get_dummies(data["race"])
C=pd.get_dummies(data["education"])
D=pd.get_dummies(data["martial-status"])
E=pd.get_dummies(data["relationship"])
F=pd.get_dummies(data["sex"])
G=pd.get_dummies(data["country"])
H=pd.get_dummies(data["income"],drop_first=True)
I=pd.get_dummies(data["occupation"])

J= data.drop(["workclass","occupation","capital-gain","capital-loss","race","education","martial-status","relationship","sex","country","income"],axis=1,inplace=True)
K=pd.concat([data,A,B,C,D,E,F,G,I,H],axis=1)

X = K.drop("H",axis=1)
y=K.iloc[:,-1] #################### PROBLEM####################


from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

print ("Accuracy of this model is ",int(100*accuracy_score(y_test,predictions)),"percent")
