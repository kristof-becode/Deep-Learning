
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential, Model, load_model #, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.layers import experimental.preprocessing.Rescaling

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os

model = load_model("MobNetv2_10epochs_fine_tuning.h5")


"""

# CHeck Alber' s chimera :-)
path = os.path.abspath('/home/becode/Downloads')
folder_path = path
image_size = (150,150)
src_file = os.path.join(path, 'customLogo.png')
img = keras.preprocessing.image.load_img(src_file, target_size=image_size)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    f"Alber's Chimera is {100 * (1 - score)} percent cat and {100 * score} percent dog."
)

"""
# CHeck 100 files in the Cats and Dogs test1 folder
path = os.path.abspath('/home/becode/LearnAI/Data/CatsnDogs/test1')
folder_path = path
image_size = (150,150)
count = 0
image_size = (150,150)
for path, dirs, files in os.walk(folder_path):
    files.sort()
    for filename in files:
        if count < 100:
            src_file = os.path.join(path, filename)
            img = keras.preprocessing.image.load_img( src_file, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            predictions = model.predict(img_array)
            predictions = tf.nn.sigmoid(predictions)
            #predictions = tf.where(predictions < 0.5, 0, 1)
            score = predictions[0]
            print(
                f"{filename} is {100 * (1 - score)} percent cat and {100 * score} percent dog."
            )
            count += 1


# CHECK on some random images from the internet
path = os.path.abspath('/home/becode/Documents/Images')
folder_path = path
image_size = (150,150)
count = 0
image_size = (150,150)
for path, dirs, files in os.walk(folder_path):
    files.sort()
    for filename in files:
            src_file = os.path.join(path, filename)
            img = keras.preprocessing.image.load_img( src_file, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            predictions = model.predict(img_array)
            score = predictions[0]
            print(
                f"{filename} is {100 * (1 - score)} percent cat and {100 * score} percent dog."
            )
            count += 1