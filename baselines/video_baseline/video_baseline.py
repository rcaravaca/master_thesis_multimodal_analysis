#!/usr/bin/env python3


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report, roc_auc_score, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from scipy import stats
import sys
import re

sys.path.append('/home/rcaravaca/Documents/Maestria/Tesis_Msc/master_thesis_multimodal_analysis/wearable_baseline')
import baselines as bs

if __name__ == "__main__":

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
	################## GET VIDEO DATASET ##################
	dataset = pd.read_csv("../dataset/dataset_video.csv")

	### VARPOS
	# dataset = dataset[['date','varpos_a','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARPOS-OHTER
	# dataset = dataset[['date','varpos_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','motion_reaction_a','motion_reaction_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### MOTION-SYNC
	dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	
	X, Y = bs.process_dataset(dataset, target_names)

	X_video_sh, y_video_sh = shuffle(X, Y)

	targets = Y.columns.values

	X = np.asarray(X_video_sh)

	#################################################


	print("\n-I- Get AUC matrics:")
	for target in targets[4:]:

		print("-I- Working on: ",target,end = '')

		y = np.asarray(y_video_sh[target])

		# classifier = KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=4, p=1, weights='uniform')
		# linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
		classifier = svm.SVC(C=2, kernel='rbf', probability=True)
		bs.get_roc_curve(classifier, X, y, target, splits=10)


	################## TRAINING MODEL ##################

	# params = {
	#             'n_neighbors': np.arange(1,50),
	#             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	#             'weights': ['uniform', 'distance'],
	#             'leaf_size': np.arange(2,50,10),
	#             'p': [1,2]
	#         }

	# scorers = {
	# 	'precision_score': make_scorer(precision_score),
	# 	'recall_score': make_scorer(recall_score),
	# 	'accuracy_score': make_scorer(accuracy_score),
	# 	'roc_auc': make_scorer(roc_auc_score)
	# }
	# grid_search_clf = bs.grid_search_wrapper(KNeighborsClassifier(), X_train, X_test, y_train['Friendly'], y_test['Friendly'] , params, scorers, refit_score='roc_auc')
		
	# results = pd.DataFrame(grid_search_clf.cv_results_)
	# results = results.sort_values(by='mean_test_roc_auc', ascending=False)
	# print(results[["params", "mean_train_roc_auc", "std_train_roc_auc", "mean_test_roc_auc", "std_test_roc_auc"]])

		# exit()





