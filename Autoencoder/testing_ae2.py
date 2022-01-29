#!/usr/bin/env python3

"""
Created on 22/01/2022

@author = Ronald
"""


from tensorflow.keras.datasets import mnist
import autoencoder as ae
from tensorflow.keras import losses

# Load example data
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()
# Scale pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
y_train = y_train.astype('float32') / 255.0
x_validation = x_validation.astype('float32') / 255.0
y_validation = y_validation.astype('float32') / 255.0
# Multimodal training data
data = [x_train, y_train]
# Multimodal validation data
validation_data = [x_validation, y_validation]
# Set network parameters
input_shapes =  {"x_data": x_train.shape[1:], "y_data": (1,)}
# Number of units of each layer of encoder network
hidden_dims = [128, 64, 8]
# Output activation functions for each modality
output_activations = ['sigmoid', 'relu']
# Name of Keras optimizer
optimizer = 'adam'
# Loss functions corresponding to a noise model for each modality
# loss = ['bernoulli_divergence', 'poisson_divergence']
# Construct autoencoder network
autoencoder = ae.multimodal_autoencoder(input_shapes, hidden_dims)

# print(autoencoder.summary())
print(autoencoder.summary())
autoencoder.compile(optimizer, loss=losses.CategoricalCrossentropy())

# # Train model where input and output are the same
autoencoder.fit(data, epochs=2, batch_size=256,
                validation_data=validation_data)