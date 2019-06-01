#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recurrent Neural Networks (RNN)
LSTM Long Sort-Term Memory
Created on Thu May 16 11:50:57 2019

@author: liam Sullivan
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

#Data preprocessing
dataset_train = pd.read_csv('AAPL-train.csv')
# Drop the Ajusted Close column
#dataset_train = dataset_train.drop('Adj Close', axis = 1)
#dataset_train2 = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Normalization
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 time stamps and 1 output
X_train = []
y_train = []
for i in range(60, 1136):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the RNN

# Initializing the RNN
regressor = Sequential()

# Adding the first layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, 
                   input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Second Layer
regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.2))

# Third Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Forth Layer
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

# Final output layer
regressor.add(Dense (units = 1))

# Making predictions and visualizing the results

# Compile
regressor.compile(optimizer = 'nadam', loss = 'mean_squared_error')

# Fit the RNN to the training set
regressor.fit(X_train, y_train, epochs = 40, batch_size = 16)

# Make predictions
dataset_test = pd.read_csv('AAPL-test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], 
                           dataset_test['Open']), 
axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60, 184):
    X_test.append(inputs[i - 60:i, 0])
    
X_test = np.array(X_test)

# Reshaping the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Apple Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time(in days)')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()















