
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
"""
# Check out both CSV's
print("-> RED WINE")
print(red_wine.shape)
print(red_wine.head(10))
print(red_wine.describe())
print(red_wine.sample(10))
print(red_wine.isna().sum())
print("-> WHITE WINE")
print(white_wine.shape)
print(white_wine.head(10))
print(white_wine.describe())
print(white_wine.sample(10))
print(white_wine.isna().sum())

# Pairplots for red and white datasets
sns.pairplot(red_wine)
plt.show()

sns.pairplot(white_wine)
plt.show()

# Pearson's coeff in correlation matrix for red and white
corr=red_wine.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(red_wine[top_features].corr(),annot=True)
plt.show()

corr=white_wine.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(white_wine[top_features].corr(),annot=True)
plt.show()


# Plot alcohol column for both datasets in one image
fig, ax = plt.subplots(1, 2)

ax[0].hist(red_wine.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white_wine.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()

# You can get the values from numpy histogram function with bins for possible values for alcohol levels
print(np.histogram(red_wine.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
print(np.histogram(white_wine.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

# Plotting Sulphates vs Quality for both red and white datasets

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red_wine['quality'], red_wine["sulphates"], color="red")
ax[1].scatter(white_wine['quality'], white_wine['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()


# Plot Alcohol - Volatile Acidity for both red and white datasets
np.random.seed(570)

redlabels = np.unique(red_wine['quality'])
whitelabels = np.unique(white_wine['quality'])

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6, 4)
whitecolors = np.append(redcolors, np.random.rand(1, 4), axis=0)

for i in range(len(redcolors)):
    redy = red_wine['alcohol'][red_wine.quality == redlabels[i]]
    redx = red_wine['volatile acidity'][red_wine.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white_wine['alcohol'][white_wine.quality == whitelabels[i]]
    whitex = white_wine['volatile acidity'][white_wine.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0, 1.7])
ax[1].set_xlim([0, 1.7])
ax[0].set_ylim([5, 15.5])
ax[1].set_ylim([5, 15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol")
# ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
# fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()

"""
## COMBINE BOTH RED AND WHITE INTO 1 DATASET - DO NOTE THAT IT IS A VERY UNBALANCED COMBINATION, 1600 red vs 4900 White
## BUT FOR THE SAKE OF THE EXPERiMENT WE'LL CONTINUE AS IS

# First add a column 'type' to each dataset; 0 for white, 1 for red
red_wine['type']= 0
white_wine['type']= 1
print(red_wine.sample(10))
print(white_wine.sample(10))

## Now we will combine the red and white datasets into 1 dataset which we will use for classification
wine = pd.concat([red_wine, white_wine], ignore_index=True) # could use append but concat is newer, ignore index of original df's
print(wine.shape)
# Now white isjust below red in one df, so for split random state is very important to split intest and train, also beacuse uneven ratios

# Pearson's from the concatenated dataset
corr = wine.corr()
top_features=corr.index
plt.figure(figsize=(20,20))
sns.heatmap(wine[top_features].corr(),annot=True)
#plt.show()

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
model.add(Dense(12, activation='relu', input_shape=(12,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

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
# Lastly, with multi-class classification, you’ll make use of categorical_crossentropy
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
SUMMARY SCORES:
[[0.]
 [1.]
 [1.]
 [1.]
 [0.]]
[0 1 1 1 0]
41/41 [==============================] - 0s 731us/step - loss: 0.0185 - accuracy: 0.9977
[0.01851234771311283, 0.9976922869682312]
Test loss: 0.01851234771311283
Test accuracy: 0.9976922869682312
confusion_matrix : [[329   3]
 [  0 968]]
precision_score : 0.9969104016477858
recall_score : 1.0
f1_score : 0.9984528107271788
cohen_kappa_score : 0.9939142755491196

"""






