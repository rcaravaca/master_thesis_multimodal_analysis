#!/usr/bin/env python3

"""
Created on 22/01/2022

@author = Ronald
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import losses

import autoencoder as ae
import datetime


(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

input_dim = 28 * 28
hidden_dim = (28 * 28) // 2
latent_dim = 64


(x_train, y_train), (x_validation, y_validation) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.0
y_train = y_train.astype('float32') / 255.0
x_validation = x_validation.astype('float32') / 255.0
y_validation = y_validation.astype('float32') / 255.0


x_train_noisy = tf.clip_by_value(x_validation, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_validation, clip_value_min=0., clip_value_max=1.)


noise_factor = 0.2
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)

x_train = x_train_noisy
# Multimodal training data
data = [x_train]
# Multimodal validation data
validation_data = [x_train_noisy]

dropout_rates = 0.15

autoencoder = ae.multimodal_autoencoder([(28, 28)], [128, 64, 16], latent_dim, dropout_rates=dropout_rates)

tf.keras.utils.plot_model(autoencoder, to_file="my_model.png", show_shapes=True, expand_nested=True, show_layer_names=True, show_layer_activations=True)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


autoencoder.fit(data,
					epochs=10,
					shuffle=True,
					batch_size=64,
					validation_data=validation_data,
					callbacks=[tensorboard_callback])

encoded_imgs = autoencoder.encoder(x_train_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_train_noisy[i])
	plt.title("original")
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i])
	plt.title("reconstructed")
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()