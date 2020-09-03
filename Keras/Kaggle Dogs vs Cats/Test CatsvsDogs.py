
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
"""
# MAKE FOLDERS 'CATS' AND 'DOGS' IN THE TRAINING FOLDER AND ADD LABELED PICTURES TO THEIR RESPECTIVE FOLDER
import os.path
import shutil
import re

dest_cats = '/home/becode/LearnAI/Data/CatsnDogs/train_sorted/cats'
dest_dogs = '/home/becode/LearnAI/Data/CatsnDogs/train_sorted/dogs'

os.makedirs(dest_cats)
os.makedirs(dest_dogs)

path = os.path.abspath('/home/becode/LearnAI/Data/CatsnDogs/train')
folder_path = path

for path, dirs, files in os.walk(folder_path):
    for filename in files:
        if re.match("dog(.)+.jpg", filename) is not None:
            src_file = os.path.join(path, filename)
            dest_file = os.path.join(dest_dogs, filename)
            shutil.copyfile(src_file, dest_file)
        if re.match("cat(.)+.jpg", filename) is not None:
            src_file = os.path.join(path, filename)
            dest_file = os.path.join(dest_cats, filename)
            shutil.copyfile(src_file, dest_file)

# Print the total number of images in each folder
print('total training cat images:', len(os.listdir('/home/becode/LearnAI/Data/CatsnDogs/train_sorted/cats')))
print('total training dog images:', len(os.listdir('/home/becode/LearnAI/Data/CatsnDogs/train_sorted/dogs')))
"""

# Create two subsets, training and validation, of the 'Cats' and 'Dogs' folders in the Training folder
# state directory to create subsets from
dir = '/home/becode/LearnAI/Data/CatsnDogs/train_sorted'
# create 'training' subset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
)
# create 'validation' subset, 20% of the total images for validation
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir,
    labels="inferred",
    label_mode="binary",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(150, 150),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
)


"""
# PLOT SOME IMAGES AND THEIR LABELS
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(2): # If I take 1 theb I get only dogs
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()
"""

# flip and rotate images to create more diversity in small dataset, 'data augmentation' !!!!
data_augmentation = keras.Sequential(
    [
       RandomFlip("horizontal"),
        RandomRotation(0.1),
    ]
)

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))


# prefetch to avoid I/O blocking
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# set params
batch_size = 64
epochs = 30
num_classes = 1 # O or 1

# define model of Convolutional Neural Network
model = Sequential()
model.add(Rescaling(1./255, input_shape=(150,150,3))) # Rescaling is set up in the model itself !!!!!
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1)) # prevents blocked and non-active RELUs
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25)) # DROPOUT layer
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))  # DROPOUT layer
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.4)) # DROPOUT layer
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3)) # DROPOUT layer
model.add(Dense(num_classes, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(model.summary())

# Train the model
model_train_dropout = model.fit(train_ds, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=val_ds)

# Save the model and the weights!
model.save("model_epochs.h5")

### Model Evaluation on validation set
# Evaluate Test set
test_eval = model.evaluate(val_ds, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Plot accuracy and loss plots
accuracy = model_train_dropout.history['accuracy']
val_accuracy = model_train_dropout.history['val_accuracy']
loss = model_train_dropout.history['loss']
val_loss = model_train_dropout.history['val_loss']
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

# This model should ideally be improved upon, it's accuracy is close to 85% but it is overfitting. A better approach
# would be to train the upper layers of a pretrained CNN.