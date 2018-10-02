
# This program is to predict the song to which year it belongs to 
# data can be downloaded from "https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD"
# Importing required libraries 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd



# Reading the dataset 

dataset = pd.read_csv('YearPredictionMSD.txt')

# drop any null values if any
dataset = dataset.dropna()

# Slicing the data according to requirement

# All ROWS and 2nd columns to last column are sliced, stored in X
X = dataset.iloc[:, 1:].values

# All ROWS and 0th column are sliced, stored in y
y = dataset.iloc[:, 0].values

# Splitting dataset in to TRAINING SET and TESTING SET
# Here, test_size=0.10018357 is used to split data exactly into -
# TRAIN DATA = 463715 and TEST DATA = 51630 as per DATA description 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10018357)

# Training and Fitting the model

multivariate = LinearRegression()
multivariate.fit(X_train, y_train)

# Predicting values using our trained model, by giveing X_test data
y_prediction= multivariate.predict(X_test)

# After prediction, Coverting integers for coparision with y_test
y_prediction.astype(int)

# Evaluating the MODEL

ex_var_score = explained_variance_score(y_test, y_prediction)
m_absolute_error = mean_absolute_error(y_test, y_prediction)
m_squared_error = mean_squared_error(y_test, y_prediction)
r_2_score = r2_score(y_test, y_prediction)

print("Explained Variance Score: "+str(ex_var_score))
print("Mean Absolute Error : "+str(m_absolute_error))
print("Mean Squared Error :"+str(m_squared_error))
print("R Squared Error :"+str(r_2_score))






