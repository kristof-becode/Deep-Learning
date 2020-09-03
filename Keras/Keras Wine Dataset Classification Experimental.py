
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model #, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

# Display options dataframes
pd.set_option('display.width',400)
pd.set_option('display.max_columns', 40)
# Display options numpy arrays
np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180

# Wine Dataset retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality

# Read CSV's for red and white wine
red_wine = pd.read_csv('/home/becode/LearnAI/Data/Wine Quality/winequality-red.csv', sep=';')
white_wine = pd.read_csv('/home/becode/LearnAI/Data/Wine Quality/winequality-white.csv', sep=';')

# First add a column 'type' to each dataset; 0 for white, 1 for red
red_wine['type']= 0
white_wine['type']= 1

## Now we will combine the red and white datasets into 1 dataset which we will use for classification
wine = pd.concat([red_wine, white_wine], ignore_index=True) # could use append but concat is newer, ignore index of original df's
## Set up train and test samples via scikit train_test_split

# X,y
X= wine.drop('type',axis=1) # features
#y= wine.type # label
y=np.ravel(wine.type) # label np.ravel 'flattens' y

# Standardize X, y already binary encoded 0,1
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

## MAKE MODEL (binary classification)

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='tanh', input_shape=(12,)))

# Add one hidden layer
model.add(Dense(8, activation='tanh'))

# Add 2 EXTRA hidden layers
model.add(Dense(20, activation='tanh'))
model.add(Dense(30, activation='tanh'))

# Add an output layer , act= sigmoid so you get probability between 0 and 1
model.add(Dense(1, activation='sigmoid'))

# Model output shape
print(model.output_shape)

# Model summary
print(model.summary())

# Model config
model.get_config()

# List all weight tensors
model.get_weights()

# Compile and fit model
#loss function = binary_crossentropy for the binary classification problem of determining whether a wine is red or white.
# With multi-class classification, you’ll make use of categorical_crossentropy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1) # in batches of 1

# Predict Values
y_pred = model.predict(X_test)
# !! IT IS PARAMOUNT TO ROUND HERE TO INT's TO CALCULATE SCORES, otherwise you
# get list of floats and you want 0's or 1's !!!
y_pred = np.round(y_pred)
print(y_pred[:5])
print(y_test[:5])

# Evaluate  model
score = model.evaluate(X_test, y_test,verbose=1)
# score is a list that holds the combination of the loss and the accuracy
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(f"confusion_matrix : {confusion_matrix(y_test,y_pred)}")
print(f"precision_score : {precision_score(y_test,y_pred)}")
print(f"recall_score : {recall_score(y_test,y_pred)}")
print(f"f1_score : {f1_score(y_test,y_pred)}")
# The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.
print(f"cohen_kappa_score : {cohen_kappa_score(y_test,y_pred)}")

"""
SUMMARY SCORES for adding more hidden layers, more hidden nodes and using tanh as activation function:
[[0.]
 [1.]
 [1.]
 [1.]
 [0.]]
[0 1 1 1 0]
41/41 [==============================] - 0s 629us/step - loss: 0.0319 - accuracy: 0.9954
[0.03194267302751541, 0.9953846335411072]
Test loss: 0.03194267302751541
Test accuracy: 0.9953846335411072
confusion_matrix : [[327   5]
 [  1 967]]
precision_score : 0.9948559670781894
recall_score : 0.9989669421487604
f1_score : 0.9969072164948455
cohen_kappa_score : 0.9878164596506136


"""





