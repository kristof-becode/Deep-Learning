
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


# flip and rotate images to create more diversity in small dataset, 'data augmentation' !!!!
data_augmentation = keras.Sequential(
    [
       RandomFlip("horizontal"),
        RandomRotation(0.2),
    ]
)

rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


#augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

conv_base = tf.keras.applications.MobileNetV2(
    input_shape=(150,150,3),
    alpha=1.0,
    include_top=False,
    weights="imagenet"
)

conv_base.trainable = False

print(conv_base.summary())
image_batch, label_batch = next(iter(train_ds))
feature_batch = conv_base(image_batch)
print(feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


#Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. You don't need an activation function here because this prediction will be treated as a logit, or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.


prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)



inputs = tf.keras.Input(shape=(150, 150, 3))
x = rescale(inputs)
x = data_augmentation(x)
x = conv_base(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

"""
inputs = keras.Input(shape=(150, 150, 3))
x = rescale(inputs)
x= data_augmentation(x)
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = conv_base(x, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
"""
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy']) #keras.metrics.BinaryAccuracy()
initial_epochs = 10
print(model.summary())
model_train = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds)

# Save the model and the weights!
model.save("MobNetv2_10epochs_feat_extract.h5")


acc = model_train.history['accuracy']
val_acc = model_train.history['val_accuracy']

loss = model_train.history['loss']
val_loss = model_train.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
#plt.show()
plt.savefig('Acc + Loss MobNetv2 feat extraction')

# FINETUNING
conv_base.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(conv_base.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in conv_base.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(1e-5), # learning rate @ default for Adam = 0.001 or 1e-3
              metrics=['accuracy'])

print(model.summary())

print(len(model.trainable_variables))

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

model_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=11,
                         validation_data=val_ds)


# Save the model and the weights!
model.save("MobNetv2_10epochs_fine_tuning.h5")


acc += model_fine.history['accuracy']
val_acc += model_fine.history['val_accuracy']

loss += model_fine.history['loss']
val_loss += model_fine.history['val_loss']



plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
#plt.show()
plt.savefig('Acc + Loss MobNetv2 fine_tune')



