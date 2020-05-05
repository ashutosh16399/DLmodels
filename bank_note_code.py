# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 04:29:38 2020

@author: Ashut
"""

# Import seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
banknotes = pd.read_csv('banknotes.xls')

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations of each class
print('Observations per class: \n', banknotes['class'].value_counts())
# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()
y = banknotes[['class']] #all columns except the last one
X = banknotes.drop(['class'],axis=1) #only the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)