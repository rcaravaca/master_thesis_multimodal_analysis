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
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class multimodal_autoencoder(Model):

	# def __init__(self, input_space, latent_space):
	def __init__(self, input_shapes, hidden_layers_dim, output_activations='linear',
				 dropout_rates=None, activity_regularizers=None,
				 **unimodal_kwargs):

		if isinstance(input_shapes, dict):
			modality_names = list(input_shapes.keys())
			input_shapes = [input_shapes[name] for name in modality_names]
		else:
			modality_names = [str(name) for name in range(len(input_shapes))]

		# make input layers by modality
		input_layers = self._make_input_layers(input_shapes, modality_names)
		fusion_shapes = self._get_output_shapes(input_layers)

		# Make concatenation layer for multimodal representation
		_concat_layer = self._concat_multimodal_input(input_layers)

		# Make multimodal encoder
		_encoder = self._create_encoder(_concat_layer, hidden_layers_dim)

		# Holding multimodal representation space
		latent_space = Input(shape=(hidden_layers_dim[-1],), name="latent_space")

		# Make multimodal decoder
		_decoder , _autoencoder = self._create_decoder(
						latent_space, _encoder, hidden_layers_dim, _concat_layer.shape[1])



		output_decoder = _decoder
		output_autoencoder = _autoencoder

		# output_decoder, output_autoencoder = self._add_activations(output_activations,
		#														   output_decoder,
		#														   output_autoencoder,
		#														   modality_names)

		output_encoder = _encoder

		super(multimodal_autoencoder, self).__init__(
						input_layers, output_autoencoder)

		self.encoder = Model(input_layers, output_encoder)
		self.decoder = Model(latent_space, output_decoder)


		# self.decoder = Model(latent_space, output_decoder)
		# self.latent_space = latent_space

		# self.encoder = tf.keras.Sequential([
		#   layers.Input(shape=(input_space,1)),
		#   layers.Dense((input_space - latent_space)/2 + latent_space, activation='relu'),
		#   layers.Dense(latent_space, activation='relu')
		# ])

		# self.decoder = tf.keras.Sequential([
		#   layers.Dense((input_space - latent_space)/2 + latent_space, activation='relu'),
		#   layers.Dense(input_space, activation='sigmoid')
		# ])

	# def call(self, x):
	# 	encoded = tf.keras.Sequential(self._encoder(x))
	# 	decoded = tf.keras.Sequential(self._decoder(encoded))
	# 	return decoded

	@staticmethod
	def _get_output_shapes(layers):

		output_shapes = [K.int_shape(layer) for layer in layers]
		return output_shapes

	@staticmethod
	def _make_input_layers(input_shapes, modality_names=None):

		if modality_names is not None:
			input_layers = [Input(shape=input_shapes[i], name=name)
							for i, name in enumerate(modality_names)]
		else:

			input_layers = [input(shape=shape) for shape in input_shapes]

		return input_layers

	@classmethod
	def _concat_multimodal_input(cls, layers):

		shapes = cls._get_output_shapes(layers)

		# In case input layers are not unidimentional need to flatten them
		flatten_layer = []
		dense_layer = []

		for i, shape in enumerate(shapes):

			if len(shape) > 1:
				# layers[i] = Flatten()(layers[i])
				flatten_layer.append(Flatten()(layers[i]))
				# dense_layer.append(Dense(shape=flatten_layer[i].dims[1].value)(flatten_layer[i]))
				
			else:
				flatten_layer.append(layers[i])

		# Making multimoldal layer
		if len(flatten_layer) > 1:
			# multimodal_layer = concatenate(dense_layer, name="concat_layer")
			input_dense_layer = []
			for flayer in flatten_layer:
				flayer_shape = cls._get_output_shapes([flayer])
				input_dense_layer.append(Dense(flayer_shape[0][1])(flayer))

			multimodal_layer = concatenate(
							input_dense_layer, name="Input_concatenation")
		else:
			multimodal_layer = layers[0]
			# concat_layer = flatten_layer[0]

		# m_shapes = cls._get_output_shapes([concat_layer])

		# multimodal_layer = Dense(
		# 				m_shapes[0][1], name="concat_layer")(concat_layer)

		return multimodal_layer

	@classmethod
	def _create_encoder(cls, input_layer, hidden_layers_dim, dropout_rates=None,
					  activity_regularizers=None):

		# kernel_constraints = cls._get_kernel_constraints(dropout_rates)
		# mm_layer = Dropout(dropout_rates[0])(mm_layer)
		encoder = input_layer

		for i, dim in enumerate(hidden_layers_dim):
			layer = Dense(dim, activation='relu',
						  name="encoder_layer_" + str(i))
			encoder = layer(encoder)
			# encoder = Dropout(dropout_rates[i+1])(encoder)
		return encoder

	@classmethod
	def _create_decoder(cls, latent_space, encoder, hidden_dims, _concat_layer_shape, 
					  fusion_shapes=None, dropout_rates=None,
					  activity_regularizers=None):

		decoder = latent_space
		autoencoder = encoder

		for i in range(len(hidden_dims) - 2, -1, -1):
			layer = Dense(hidden_dims[i], activation='relu',
						  name="decoder_layer_" + str(i))
			# dropout = Dropout(0.0)
			decoder = layer(decoder)
			autoencoder = layer(autoencoder)

		decoder = Dense(_concat_layer_shape, name="concat_output_layer")(decoder)
		autoencoder = Dense(_concat_layer_shape, name="concat_output_layer")(autoencoder)

		return decoder , autoencoder
		# fusion_decoder = [None] * len(fusion_shapes)
		# fusion_autoencoder = [None] * len(fusion_shapes)
		# for i, shape in enumerate(fusion_shapes):
		#   # No activation and no dropout in unimodal separation layers
		#   layer = Dense(np.prod(shape[1:]))
		#   fusion_decoder[i] = Reshape(shape[1:])(layer(decoder))
		#   fusion_autoencoder[i] = Reshape(shape[1:])(layer(autoencoder))

		# return fusion_decoder, fusion_autoencoder

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

	def compile(self, optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
				loss_weights=None, sample_weight_mode=None,
				target_tensors=None):
		"""
		Sets the model configuration for training.
		Parameters
		----------
		optimizer : str, optional
			Name of optimization algorithm.  (Default: 'adam')
		loss : str, callable, dict or list, optional
			Loss functions, including Bregman divergences, for each modality.
			(Default: 'gaussian_divergence')
		loss_weights : dict or list of floats, optional
			Loss weights for each modality.  `None` corresponds to weight `1.0`
			for each modality.  (Default: `None`)
		sample_weight_mode : str, list or dict, optional
			Sample weight mode for each modality.  Each mode can be `None`
			corresponding to sample-wise weighting or 'temporal' for
			timestep-wise weighting.  (Default: `None`)
		target_tensors : tensor, list of tensors or dict, optional
			Target tensors to be used instead of `data` arguments for training.
			(Default: `None`)
		"""
		# Replace Bregman divergence strings with actual functions
		# loss = self._replace_bregman_strings(loss)
		# For dicts, rename keys to match output keys
		loss = self._rename_output_keys(loss)
		loss_weights = self._rename_output_keys(loss_weights)
		sample_weight_mode = self._rename_output_keys(sample_weight_mode)
		target_tensors = self._rename_output_keys(target_tensors)
		super(multimodal_autoencoder,
			  self).compile(optimizer=optimizer, loss=loss,
							loss_weights=loss_weights,
							sample_weight_mode=sample_weight_mode,
							target_tensors=target_tensors)

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
