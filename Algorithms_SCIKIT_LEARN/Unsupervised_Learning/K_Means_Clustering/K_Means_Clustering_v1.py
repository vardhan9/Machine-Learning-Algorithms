#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:40:16 2018

@author: rajavardhan
"""
#https://stackabuse.com/k-means-clustering-with-scikit-learn/
##############################

#importing libraries
import matplotlib.pyplot as plt  
#%matplotlib inline
import numpy as np 
import pandas as pd 
 

## Reading the data
train = pd.read_csv("Iris.csv")

#Slicing the columns
f1 = train['SepalLengthCm'].values
f2 = train['SepalWidthCm'].values

f3 = train['PetalLengthCm'].values
f4 = train['PetalWidthCm'].values

# Converting All sliced columns into an ARRAY

X = np.array(list(zip(f1, f2,f3,f4)))

#plotting an ARRAY before Clustering
plt.scatter(f1,f2,f3,f4,label='True Position')
plt.show()

# import k-means Clustering model
from sklearn.cluster import KMeans

# Giving number of clusters
kmeans = KMeans(n_clusters=3)  

#Fitting a model
kmeans.fit(X)

# Printing CENTROIDS OF CLUSTERS
print(kmeans.cluster_centers_)  

#printing LABELS of the data (defining whether it belongs to Cluster 1 or 2 or 3)
print(kmeans.labels_)  
plt.scatter(f1,f2,f3,f4, cmap='rainbow')
plt.show()
