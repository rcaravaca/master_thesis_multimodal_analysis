#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score, plot_roc_curve
from sklearn.utils import shuffle
from scipy import stats
import sys


sys.path.append('/home/rcaravaca/Documents/Maestria/Tesis_Msc/master_thesis_multimodal_analysis/baselines/baselines_functions.py')
import baselines_utils as bu
import ensemble_classifier as ec


def baselines_excecution():

	print("\n######################################################")
	print("\n\t\tVIDEO BASELINE\n")
	print("######################################################\n")

	target_names = {"M_3": "SeeAgain",
					"M_4": "Friendly",
					"M_5": "Sexual",
					"M_6": "Romantic",
					"PP_M_3": "PP_SeeAgain",
					"PP_M_4": "PP_Friendly",
					"PP_M_5": "PP_Sexual",
					"PP_M_6": "PP_Romantic"
				}

	# date	avg_difangle	var_difangle	avgdis	vardis	decardis	movdistr_mean_a	movdistr_var_a	movdistr_mean_b	movdistr_var_b	motionsync_a_1	motionsync_a_2	motionsync_a_3	motionsync_a_4	motion_reaction_a	motion_reaction_b	varpos_a	varpos_b	M_1	M_2	M_3	M_4	M_5	M_6	PP_M_1	PP_M_2	PP_M_3	PP_M_4	PP_M_5	PP_M_6
	################## GET VIDEO DATASET ############
	dataset = pd.read_csv("../dataset/dataset_video.csv")

	# print(dataset.head(10))
	sex = '_F'
	dataset = dataset[dataset['date'].str.contains(sex)]
	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)
	# print(dataset.head(10))

	# exit()
	### VARPOS
	# dataset = dataset[['date','varpos_a','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARPOS-OHTER
	# dataset = dataset[['date','varpos_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARDIS
	dataset = dataset[['date','vardis','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','motion_reaction_a','motion_reaction_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### MOTION-SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	X_video, Y_video = bu.process_dataset(dataset, target_names)

	#################################################

	################## GET WEARABLE DATASET #########
	dataset = pd.read_csv("../dataset/dataset_wearables.csv")

	dataset = dataset[dataset['date'].str.contains(sex)]
	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)

	dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='sym')))]

	X_wearable, Y_wearable = bu.process_dataset(dataset, target_names)

	#################################################
	Xv_sh, Yv_sh, Xw_sh, Yw_sh = shuffle(X_video, Y_video, X_wearable, Y_wearable)
	targets = Y_wearable.columns.values

	Xv = np.asarray(Xv_sh)
	Xw = np.asarray(Xw_sh)

	#################################################
	knn_video = KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=3, p=1, weights='uniform')
	lr_wearable = LogisticRegression(C=0.00001, max_iter=100000, solver='liblinear')
	svm_video = svm.SVC(C=2, kernel='rbf', probability=True)
	majority_vote = ec.majority_vote(clfs=[svm_video, lr_wearable])
	#################################################

	cv = StratifiedKFold(n_splits=10)
	#################################################

	mean_auc_v = std_auc_v = predict_probas_v = 0
	mean_auc_w = std_auc_w = predict_probas_w = 0
	mean_auc_mv = std_auc_mv = 0

	print("\n-I- Get AUC matrics:")
	print("Target, Video (AUC), Wearables (AUC), Majority_vote (AUC)")
	for target in targets:

		print(target, end="")

		Yv = np.asarray(Yv_sh[target])
		Yw = np.asarray(Yw_sh[target])
		splits = cv.split(Xv, Yv)

		trains = []
		tests = []
		for i , (train, test) in enumerate(splits):
			trains.append(train)
			tests.append(test)

		mean_auc_v, std_auc_v, predict_probas_v = bu.get_roc_curve(svm_video, Xv, Yv, target, trains, tests, "video")
		mean_auc_w, std_auc_w, predict_probas_w = bu.get_roc_curve(lr_wearable, Xw, Yw, target, trains, tests, "wearables")
		mean_auc_mv, std_auc_mv = bu.majority_vote(predict_probas_v, predict_probas_w, tests, Yv, target, "majority_vote")

		print(', %0.2f +- %0.2f, %0.2f +- %0.2f, %0.2f +- %0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w, mean_auc_mv, std_auc_mv))


if __name__ == "__main__":

	baselines_excecution()