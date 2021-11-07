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
import baselines as bs

if __name__ == "__main__":

	print("\n######################################################")
	print("\n\t\tWEARABLE BASELINE\n")
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

	################## GET DATASET ##################
	dataset = pd.read_csv("../dataset/dataset_wearables.csv")

	X, Y = bs.process_dataset(dataset, target_names)
	X_wearable_sh, y_wearable_sh = shuffle(X, Y)

	targets = Y.columns.values

	X = np.asarray(X_wearable_sh)
	#################################################

	print("\n-I- Get AUC matrics:")
	for target in targets[4:]:

		print("-I- Working on: ",target,end = '')

		y = np.asarray(y_wearable_sh[target])
		
		classifier = LogisticRegression(C=0.00001, max_iter=100000, solver='liblinear')
		bs.get_roc_curve(classifier, X, y, target, splits=10)

		################## TRAINING MODEL ##################

		# params = {
		#             'C': np.arange(0,2,0.1),
		#             'max_iter': [10000],
		#             'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
		#         }

		# scorers = {
		# 	'precision_score': make_scorer(precision_score),
		# 	'recall_score': make_scorer(recall_score),
		# 	'accuracy_score': make_scorer(accuracy_score),
		# 	'roc_auc': make_scorer(roc_auc_score)
		# }
		# grid_search_clf = bs.grid_search_wrapper(LogisticRegression(), X_train, X_test, y_train, y_test , params, scorers, refit_score='roc_auc')
		
		# results = pd.DataFrame(grid_search_clf.cv_results_)
		# results = results.sort_values(by='mean_test_roc_auc', ascending=False)
		# print(results[["param_C", "params", "mean_train_roc_auc", "std_train_roc_auc", "mean_test_roc_auc", "std_test_roc_auc"]])

		# exit()


