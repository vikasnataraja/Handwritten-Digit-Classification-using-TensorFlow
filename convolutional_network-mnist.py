# future is imported to allow the use of different versions of Python
from __future__ import division, print_function, absolute_import
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
from random import randint

# Keras is a deep learning that is used to build the neural network
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
num_classes = 10           # we have 10 digits so we need 10 classes

train_images = train_images.astype('float') / 255. # normalize pixel values
train_images = np.expand_dims(train_images, -1) # add a channel dimension as this is what Keras expects
test_images = test_images.astype('float') / 255.
test_images = np.expand_dims(test_images, -1)

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels  = keras.utils.to_categorical(test_labels, num_classes)

print('Number of training images: ',train_images.shape[0])
print('Each training image is of size: ',train_images.shape[1:])
print('Number of training labels: ',train_labels.shape[0],'\n')
print('Number of test images: ',test_images.shape[0])
print('Each test image is of size: ',test_images.shape[1:])
print('Number of test labels: ',test_images.shape[0])

model_dir = os.getcwd()

## define architecture details
input_shape = (28, 28, 1)     # because our images are 28 x 28 pixels across
num_filters = [16, 32, 64] # filters for convolution => it's usually a good idea to double the number of filters in each step
kernel_size = (3, 3)       # this is the convolution kernel
num_epochs  = 10         # number of epochs to train for\
learning_rate = 0.001      # learning rate for the model weights. recommended to start with a low number like 0.0001
batch_size = 16           # batch size of the training set

# Create the neural network

def model_architecture(input_shape:tuple, num_filters:list, kernel_size:tuple, num_classes:int):
    model = keras.Sequential(
    [
        Input(shape=input_shape),
        Conv2D(num_filters[0], kernel_size=kernel_size, activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(num_filters[1], kernel_size=kernel_size, activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(num_filters[2], kernel_size=kernel_size, activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"), # we want probability as the output
    ])

    return model

# Build the model
model = model_architecture(input_shape, num_filters, kernel_size, num_classes)

# callbacks to the model
# 1. stop the model early if it is not learning anything
# 2. save the model (checkpoint) only if the loss has improved
callbacks = [keras.callbacks.EarlyStopping(patience=50, verbose=1),
             keras.callbacks.ModelCheckpoint(filepath='cnn_model.h5', monitor="val_loss", save_best_only=True, verbose=1)]

# our cost function or loss function is cross-entropy which is probabalistic
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # fit the model to the dataset X: train_images, y: train_labels
history = model.fit(train_images, train_labels,
          batch_size=batch_size, epochs=num_epochs,
          validation_split=0.1, verbose=1,
          callbacks=callbacks)


fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# first plot will be of accuracies
ax[0].plot(history.history['accuracy'], color='blue', label='training')
ax[0].plot(history.history['val_accuracy'], color='orange', label='validation')
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epoch')
ax[0].legend()

# second plot will be of the errors
ax[1].plot(history.history['loss'], color='blue', label='training')
ax[1].plot(history.history['val_loss'], color='orange', label='validation')
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].legend(['training', 'validation'])

fig.savefig('train_test.png', dpi=200)

plt.close()

score = model.evaluate(test_images, test_labels, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# Use the model to predict the images class
predictions = model.predict(test_images)

# Display
fig = plt.figure(figsize=(16, 16))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(test_images[i],cmap='gray')
    plt.title('Predicted label = {}'.format(np.argmax(predictions[i])))
    plt.axis('off')

fig.savefig('predictions.png', dpi=200)
plt.close()
# plt.show()

