#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:54:24 2018

@author: rajavardhan
"""
################# NOT IN DETAIL ###############################
# This model is to predict " SURVIVED or NOT in TITANIC .

import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
b=data.info()
d=data.isnull().sum()
g=data.drop("Cabin",axis=1,inplace=True)
k=data.dropna(inplace=True)
l=pd.get_dummies(data["Sex"],drop_first=True)
m=pd.get_dummies(data["Embarked"],drop_first=True)
n=pd.get_dummies(data["Pclass"],drop_first=True)
o= pd.concat([data,l,m,n],axis=1)
p=o.drop(["Sex","Name","Pclass","Embarked","Ticket","PassengerId"],axis=1,inplace=True)


X = o.drop("Survived",axis=1)
y= o["Survived"]

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