#!/usr/bin/env python3

"""
Created on 02/10/2022

@author = Ronald
"""



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
import datetime
from scipy import stats


import sys
sys.path.append('../Autoencoder')
import autoencoder as ae


def normalizing(dataset):

	for feature in dataset.columns:
		dataset[feature] = stats.zscore(np.asarray(dataset[feature]))

	return dataset

if __name__ == "__main__":


	## video training data
	video_train = pd.read_csv("../dataset/dataset_video_days_1_2.csv")
	video_test = pd.read_csv("../dataset/dataset_video_days_3.csv")

	## wearable training data
	wearable_train = pd.read_csv("../dataset/dataset_wearables_days_1_2.csv")
	wearable_test = pd.read_csv("../dataset/dataset_wearables_days_3.csv")

	## get targets
	targets_train = video_train.filter(regex='^M_?').columns
	targets_PP_train = video_train.filter(regex='PP_M_?').columns

	## dropping targets from video data
	video_train = video_train.drop(targets_train, axis=1)
	video_train = video_train.drop(targets_PP_train, axis=1)
	video_test = video_test.drop(targets_train, axis=1)
	video_test = video_test.drop(targets_PP_train, axis=1)

	## dropping targets from wearablw data
	wearable_train = wearable_train.drop(targets_train, axis=1)
	wearable_train = wearable_train.drop(targets_PP_train, axis=1)
	wearable_test = wearable_test.drop(targets_train, axis=1)
	wearable_test = wearable_test.drop(targets_PP_train, axis=1)

	## data concatenation
	train = pd.concat([video_train, wearable_train.reindex(video_train.index)], axis=1)
	test = pd.concat([video_test, wearable_test.reindex(video_test.index)], axis=1)

	## dropping date column
	train = train.drop("date", axis=1)
	test = test.drop("date", axis=1)

	## normalizing
	train = normalizing(train)
	test = normalizing(test)

	train_shape = train.shape
	train = train.astype('float32')
	test = test.astype('float32')

	# print(train.head())

	# exit()

	dropout_rates = 0

	hidden_dims = [train_shape[1]/2, train_shape[1]/4, train_shape[1]/8, train_shape[1]/16]
	
	####################
	autoencoder = ae.multimodal_autoencoder([(train_shape[1],)], hidden_dims, dropout_rates=dropout_rates)
	###################
	# autoencoder = ae.Autoencoder(train_shape[1], train_shape[1]/2, train_shape[1]/8)

	#################

	tf.keras.utils.plot_model(autoencoder, to_file="autoencoder.png", show_shapes=True, expand_nested=True, show_layer_names=True, show_layer_activations=True)


	print(autoencoder.summary())

	# exit()
	autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	autoencoder.fit(train, train,
                epochs=10,
                shuffle=True,
                batch_size=2,
                # callbacks=[tensorboard_callback],
                validation_data=(test, test))
