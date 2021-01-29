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


from scipy import stats
import baselines as bs

if __name__ == "__main__":

	
	target_names = {"M_3": "SeeAgain",
					"M_4": "Friendly",
					"M_5": "Sexual",
					"M_6": "Romantic"
				}

	################## TRAINING DATASET ##################
	dataset = pd.read_csv("../dataset/dataset_werables__20_01_2021__19_57_14_aux.csv")
	dataset = dataset.rename(columns=target_names)

	features = dataset.columns.values[1:-12]

	## replacing outlier values in targets
	for key in target_names.keys():
		dataset = bs.replace_values(dataset, 999, [target_names.get(key)])

	# Z-Score Normalizing for all features
	for feature in features:
		dataset[feature] = stats.zscore(np.asarray(dataset[feature]))

	dataset_full = dataset
	dataset = dataset.drop(['date', 'M_1', 'M_2', 'SeeAgain', 'Friendly', 'Sexual', 'Romantic', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6'], axis=1)

	targets = ['SeeAgain', 'Friendly', 'Sexual', 'Romantic']

	target_col = bs.binarize_target(dataset_full,targets,False)
	X_train, X_test, y_train, y_test = train_test_split(dataset, target_col, test_size=0.2)
	
	X = np.asarray(X_train)

	for key in target_names.keys():
		target = target_names.get(key)

		y = np.asarray(y_train[target])

		# print(X_train)

		# ################## TESTING DATASET ##################
		# dataset = pd.read_csv("../dataset/dataset_werables__20_01_2021__19_57_14_test.csv")

		# features = dataset.columns.values[1:-12]
		# ## replacing values
		# dataset = bs.replace_values(dataset, 999, target)

		# # Z-Score Normalizing for all features
		# for feature in features:
		# 	dataset[feature] = stats.zscore(np.asarray(dataset[feature]))

		# target_col = bs.binarize_target(dataset,target[0],False)
		# dataset = dataset.drop(['date', 'M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6'], axis=1)
		
		# X_test = dataset
		# y_test = target_col


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
		
		random_state = np.random.RandomState(0)
		classifier = LogisticRegression(C=2, max_iter=10000, solver='newton-cg')
		bs.get_roc_curve(classifier, X, y, target, splits=10)


