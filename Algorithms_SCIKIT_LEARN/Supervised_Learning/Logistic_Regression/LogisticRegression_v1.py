#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 18:19:18 2018

@author: rajavardhan
"""
# Here is the detailed explanations of this model " LOGISTIC REGRESSION "
#####---------->READING THE DATA<--------------------------------------########

import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
#print (data)


#-------------> ANALYSING THE DATA<---------------------

import matplotlib.pyplot as plt
import seaborn as sn

# bar grap to show number of people survived and not survived
#a=sn.countplot(x="Survived",data =data)

# bar grap to show number of people survived and not survived wrt pclass
# X,hue can be changed arrocding to requirement
#sn.countplot(x="survived",hue="pclass",data =data)

# plotting HISTOGRAM
# can be plotted for different INDEXES

#data["pclass"].hist()

# for data information 
b=data.info()

#----------------> DATA WRANGLING <------------------------------

# check null values in dataset
c=data.isnull()
# check total number of null values in dataset
d=data.isnull().sum()

# plotting HEAT MAP to view null values

#e=sn.heatmap(data.isnull(),yticklabels=False)

# Plotting BOX PLOT
#f=sn.boxplot(x="Pclass",y="Age",data=data)

# dropping INDEXES that has more null values
g=data.drop("Cabin",axis=1,inplace=True)

#h=data.drop("boat",axis=1,inplace=True)
#i=data.drop("body",axis=1,inplace=True)
#j=data.drop("home.dest",axis=1,inplace=True)

# drop all null values from the data
k=data.dropna(inplace=True)

## ALL the unnecessary and null are values are deleted

# replacing CATEGORICAL data with 0 and 1

# converting male as 1 and female as 0, dropping on coloumn beacuse 
#if not male , the there should be a female and vice versa

l=pd.get_dummies(data["Sex"],drop_first=True)
m=pd.get_dummies(data["Embarked"],drop_first=True) # if both are 0 , then they belong to other destination


# if both are 0 , then they belong to other class
n=pd.get_dummies(data["Pclass"],drop_first=True)

# Adding converted columns to the data

o= pd.concat([data,l,m,n],axis=1)

# DOUBT HERE
#dropping original indexes that are converted to 0 and 1

p=o.drop(["Sex","Name","Pclass","Embarked","Ticket","PassengerId"],axis=1,inplace=True)

##----------------> TRAIN DATA <------------------
X = o.drop("Survived",axis=1)
y= o["Survived"]

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

# Calculating accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

print ("Accuracy of this model is ",int(100*accuracy_score(y_test,predictions)),"percent")
#### INPUT values for prediction ##########

#X_test1=[[50,0,0,9,1,0,1,0,0]] # give input values here
#predictions = logmodel.predict(X_test1)
#print(predictions)

