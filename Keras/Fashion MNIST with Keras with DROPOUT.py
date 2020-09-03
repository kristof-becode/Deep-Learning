
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model #, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print(f'Training data shape X,y : {X_train.shape} , {y_train.shape}')
print(f'Testing data shape X,y :  {X_test.shape}, {y_test.shape}')
print('Training data type : ', X_train.dtype, y_train.dtype)
print('Testing data type : ', X_test.dtype, y_test.dtype)

# Find the unique numbers from the train labels
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

### Plot first image of test and train set
# Display the first image in training data
plt.subplot(121)
plt.imshow(X_train[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))
#plt.show()

# Display the first image in testing data
plt.subplot(122)
plt.imshow(X_test[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_test[0]))
#plt.show()

### Preprocess data
#As a first step, convert each 28 x 28 image of the train and test set into a matrix of size 28 x 28 x 1
# which is fed into the network.
X_train = X_train.reshape(-1, 28,28, 1)
X_test = X_test.reshape(-1, 28,28, 1)
print(X_train.shape, X_test.shape)

# The data right now is in an int8 format, so before you feed it into the network you need to convert its type
# to float32, and you also have to rescale the pixel values in range 0 - 1 inclusive. So let's do that!
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

# Change the labels from categorical to one-hot encoding
y_train_OH = to_categorical(y_train)
y_test_OH = to_categorical(y_test)
print('Original label:', y_train[0])
print('After conversion to one-hot:', y_train_OH[0])

# Split TRAINING data in training and validation data (80/20) !!!! Divide train in train/validation !!!!
X_train,X_valid,label_train,label_valid = train_test_split(X_train, y_train_OH, test_size=0.2, random_state=13)
print(X_train.shape,X_valid.shape,label_train.shape,label_valid.shape)

### Model the data
""""        !!!!! WITH DROPOUT LAYERS !!!!!
You can add a dropout layer to overcome the problem of overfitting to some extent.
Dropout randomly turns off a fraction of neurons during the training process, 
reducing the dependency on the training set by some amount
"""

# You will use a batch size of 64 using a higher batch size of 128 or 256 is also preferable
# it all depends on the memory.
batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1)) # prevents blocked and non-active RELUs
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25)) # DROPOUT
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))  # DROPOUT
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4)) # DROPOUT
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3)) # DROPOUT
fashion_model.add(Dense(num_classes, activation='softmax')) # softmax activation function with 10 units for our classes

"""
Save the model and the weights!
"""
fashion_model.save("fashion_model_dropout.h5py")

# Compile
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(fashion_model.summary())

### Train the model
fashion_train_dropout = fashion_model.fit(X_train, label_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, label_valid))

"""
Save the model and the weights!
"""
fashion_model.save("fashion_model_dropout.h5py")

### Model Evaluation on Test set
# Evaluate Test set
test_eval = fashion_model.evaluate(X_test, y_test_OH, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Plot accuracy and loss plots
accuracy = fashion_train_dropout.history['accuracy']
val_accuracy = fashion_train_dropout.history['val_accuracy']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


## Predict Labels

predicted_classes = fashion_model.predict(X_test)
# Since the predictions you get are floating point values, it will not be feasible to compare the
# predicted labels with true test labels. So, you will round off the output which will convert the float values
# into an integer. Further, you will use np.argmax() to select the index number which has a higher value in a row.

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

print(predicted_classes.shape, y_test.shape)

# Correct ones
correct = np.where(predicted_classes==y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    plt.tight_layout()
plt.show()

# Incorrect ones
incorrect = np.where(predicted_classes!=y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.tight_layout()
plt.show()
## Classification Report

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))


