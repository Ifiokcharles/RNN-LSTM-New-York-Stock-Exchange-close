# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 21:35:01 2019

@author: charles pc
"""
# Part 1 - Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('prices.csv')
dataset = pd.read_csv('prices.csv', nrows=1500)
dataset_drop = dataset.drop('symbol', axis=1, inplace=True)
close_dataset = dataset.iloc[:,2:3].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
dataset_train_close, dataset_test_close, training_set_close, test_set_close = train_test_split(dataset, close_dataset , test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import MinMaxScaler #we use minmax bcos we are dealing with RNN
sc = MinMaxScaler(feature_range = (0,1)) #all the scaled stock prices will be between 0 and 1
training_set_scaled = sc.fit_transform(training_set_close)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1200): # we start from the 60th stoke price of our dataset i.e 
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0]) #we want to predict the stock price at i+1 i.e 60
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping #we use reshape to add another dimensionality
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor_RNN = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor_RNN.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor_RNN.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor_RNN.add(LSTM(units = 50, return_sequences = True))
regressor_RNN.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor_RNN.add(LSTM(units = 50, return_sequences = True))
regressor_RNN.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor_RNN.add(LSTM(units = 50))
regressor_RNN.add(Dropout(0.2))

# Adding the output layer
regressor_RNN.add(Dense(units = 1))

# Compiling the RNN
regressor_RNN.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor_RNN.fit(X_train, y_train, epochs = 150, batch_size = 32)

# Part 3 - Making the predictions and visualising the results


# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train_close['close'], dataset_test_close['close']), axis = 0) #axis=0 is vertical concat
inputs = dataset_total[len(dataset_total) - len(dataset_test_close) - 60:].values
inputs = inputs.reshape(-1,1)#to convert to numoy array
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 360):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor_RNN.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_set_close, color = 'red', label = 'Real New York Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted New Stock Price')
plt.title('New York Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('New York Stock Price')
plt.legend()
plt.show()
