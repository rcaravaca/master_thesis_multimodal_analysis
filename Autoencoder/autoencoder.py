#!/usr/bin/env python3

"""
Created on 11/27/2021

@author = Ronald
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import losses
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras import losses

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import Model
import tensorflow.keras.backend as K

class multimodal_autoencoder(Model):

	# def __init__(self, input_space, latent_space):
	def __init__(self, input_shapes, hidden_layers_dim, output_activations='linear',
				 dropout_rates=None, activity_regularizers=None,
				 **unimodal_kwargs):

		
		super(multimodal_autoencoder, self).__init__()

		if isinstance(input_shapes, dict):
			modality_names = list(input_shapes.keys())
			input_shapes = [input_shapes[name] for name in modality_names]
		else:
			modality_names = [str(name) for name in range(len(input_shapes))]

		if dropout_rates is None:
			# No dropout if no rates specified
			dropout_rates = [0.0] * len(hidden_layers_dim)
		else:
			if np.isscalar(dropout_rates):
				dropout_rates = [dropout_rates] * len(hidden_layers_dim)

		# tf.keras.regularizers.L2(0.01)
		if not isinstance(activity_regularizers, list):
			activity_regularizers = [activity_regularizers] \
				* (len(hidden_layers_dim)+1)

		# make input layers by modality
		input_models = self._create_input_layers(input_shapes, modality_names)
		
		# # Make concatenation layer for multimodal representation
		concat_layer = self._concat_multimodal_input(input_models)
		
		# # Make multimodal encoder
		_encoder = self._create_encoder(concat_layer, hidden_layers_dim, dropout_rates=dropout_rates,
											 activity_regularizers=activity_regularizers)
		
		# # Holding multimodal representation space
		# latent_space = Input(shape=(hidden_layers_dim[-1],), name="latent_space")

		# # Make multimodal decoder
		_decoder = self._create_decoder(
						_encoder, hidden_layers_dim, input_shapes, dropout_rates=dropout_rates, 
												activity_regularizers=activity_regularizers)

		super(multimodal_autoencoder, self).__init__(
						[model.input for model in input_models], _decoder)

		self.encoder = Model(inputs=[model.input for model in input_models], outputs=_encoder)
		self.decoder = Model(inputs=_encoder, outputs=_decoder)


	# def call(self, x):

	# 	encoded = self.encoder(x)
	# 	decoded = self.decoder(encoded)
	# 	return decoded

	@staticmethod
	def _get_output_shapes(layers):

		output_shapes = [K.int_shape(layer) for layer in layers]
		return output_shapes

	@staticmethod
	def _create_input_layers(input_shapes, modality_names=None):

		input_models = []

		if modality_names is not None:

			for i, name in enumerate(modality_names):
				inpt = Input(shape=input_shapes[i], name=str(name)+"_Input")
				layers = Flatten(name=str(name)+"_Flatten")(inpt)
				layers = Dense(units=layers.shape.dims[1].value, name=str(name)+"_Dense")(layers)
				layers = Model(inputs=inpt, outputs=layers)
				input_models.append(layers)

		else:
			for i, shape in enumerate(input_shapes):
				inpt = Input(shape=shape)
				layers = Flatten()(inpt)
				layers = Dense(units=layers.shape.dims[1].value)(layers)
				layers = Model(inputs=inpt, outputs=layers)
				input_models.append(layers)

		return input_models

	@classmethod
	def _concat_multimodal_input(cls, input_models):

		model_outputs = [model.output for model in input_models]
		multimodal_layer = concatenate(model_outputs)

		return multimodal_layer

	@classmethod
	def _create_encoder(cls, input_layer, hidden_layers_dim, dropout_rates=None,
					  activity_regularizers=None):

		encoder = input_layer

		for i, dim in enumerate(hidden_layers_dim):
			encoder = Dense(dim, activation='relu', 
								name="encoder_" + str(i),
								activity_regularizer=activity_regularizers[i])(encoder)
			encoder = Dropout(dropout_rates[i])(encoder)

		return encoder

	@classmethod
	def _create_decoder(cls, latent_space, hidden_dims, input_shapes,
					  fusion_shapes=None, dropout_rates=None,
					  activity_regularizers=None):

		decoder = latent_space

		for i in range(len(hidden_dims) - 2, -1, -1):
			decoder = Dense(hidden_dims[i], 
							activation='relu',
						  	name="decoder_" + str(i), 
						  	activity_regularizer=activity_regularizers[i])(decoder)
			decoder = Dropout(dropout_rates[i])(decoder)
		
		final_layers = []
		for i, shape in enumerate(input_shapes):
			layers = Dense(np.prod(np.asarray(shape)), name="output"+str(i))(decoder)
			layers = Reshape(shape)(layers)
			final_layers.append(layers)


		return final_layers

	@classmethod
	def _add_activations(cls, output_activations, output_decoder,
						 output_autoencoder, modality_names=None):

		output_activations = [output_activations] * len(output_decoder)

		activations = [Activation(activation)
					   for activation in output_activations]

		if modality_names is None:

			activations = [Activation(output_activations[i],
									  name=cls._append_output_name(name))
						   for i, name in enumerate(modality_names)]

		for i, activation in enumerate(activations):
			output_decoder[i] = activation(output_decoder[i])
			output_autoencoder[i] = activation(output_autoencoder[i])

		return output_decoder, output_autoencoder

	@classmethod
	def _rename_output_keys(cls, structure):
		"""
		Checks whether the argument is a dict and if so, appends output labels
		to its keys.
		Parameters
		----------
		structure : object
			An input object that is modified if it is a dict.
		Returns
		-------
		output_structure : object
			Either the unmodified argument or a dict with the same values and
			keys with output labels appended.
		"""
		if isinstance(structure, dict):
			return {cls._append_output_name(key): value
					for key, value in structure.items()}
		return structure


	def fit(self, data=None, batch_size=None, epochs=1, verbose=1,
			callbacks=None, validation_split=0.0, validation_data=None,
			shuffle=True, sample_weight=None, validation_sample_weight=None,
			initial_epoch=0, steps_per_epoch=None, validation_steps=None):

		target_data = self._rename_output_keys(data)
		if validation_data is not None:
			validation_target_data = self._rename_output_keys(validation_data)
			if validation_sample_weight is None:
				validation_data = (validation_data, validation_target_data)
			else:
				validation_data = (validation_data, validation_target_data,
								   validation_sample_weight)
		return super(multimodal_autoencoder,
					 self).fit(x=data, y=target_data, batch_size=batch_size,
							   epochs=epochs, verbose=verbose,
							   callbacks=callbacks,
							   validation_split=validation_split,
							   validation_data=validation_data,
							   shuffle=shuffle,
							   sample_weight=sample_weight,
							   initial_epoch=initial_epoch,
							   steps_per_epoch=steps_per_epoch,
							   validation_steps=validation_steps)

	@classmethod
	def _rename_output_keys(cls, structure):
		"""
		Checks whether the argument is a dict and if so, appends output labels
		to its keys.
		Parameters
		----------
		structure : object
			An input object that is modified if it is a dict.
		Returns
		-------
		output_structure : object
			Either the unmodified argument or a dict with the same values and
			keys with output labels appended.
		"""
		if isinstance(structure, dict):
			return {cls._append_output_name(key): value
					for key, value in structure.items()}
		return structure
class Autoencoder(Model):

	def __init__(self, input_dim, hidden_dim, latent_dim):

		super(Autoencoder, self).__init__()

		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.latent_dim = latent_dim

		self.encoder = tf.keras.Sequential([
						Flatten(),
						Dense(input_dim, activation='sigmoid'),
						Dense(hidden_dim, activation='sigmoid'),
						Dense(latent_dim, activation='relu'),
		])
		self.decoder = tf.keras.Sequential([
						Dense(latent_dim, activation='relu'),
						Dense(hidden_dim, activation='sigmoid'),
						Dense(input_dim, activation='sigmoid'),
						Reshape((28, 28))
		])

	def call(self, x):

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
