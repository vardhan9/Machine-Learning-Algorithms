# This is to predict housing prices
import pandas as pd
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt
training = pd.read_csv('train_boston_housing.csv')
x_train=training.iloc[:,1]
y_train=training.iloc[:,14]

x_train_array=np.array(x_train)
y_train_array=np.array(y_train)

testing = pd.read_csv('test_boston_housing.csv')
x_test=testing.iloc[:,1]
y_test=testing.iloc[:,13]

x_test_array=np.array(x_test)
y_test_array=np.array(y_test)



print(x_train_array)
print(y_train_array)

slope,intercept,r_value,p_value,stderr = linregress(x_train_array,y_train_array)

y_predict =[slope*i + intercept for i in x_test_array]

plt.plot(x_train_array,y_train_array,'o')
plt.plot(x_test_array,y_predict)
plt.xlabel("crime rate")
plt.ylabel("housing price")
