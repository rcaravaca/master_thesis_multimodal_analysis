#!/usr/bin/env python3


import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import sys
import ensemble_classifier as ec
from sklearn.utils import shuffle
from sklearn import svm
import pandas as pd

sys.path.append('/home/rcaravaca/Documents/Maestria/Tesis_Msc/master_thesis_multimodal_analysis/baselines/baselines_functions.py')
import baselines_functions as bs

# def get_voting():
# 	models = list()
# 	models.append('knn_video', KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=3, p=1, weights='uniform'))
# 	models.append('lr_wearable', LogisticRegression(C=2, max_iter=10000, solver='newton-cg'))
# 	ensemble = VotingClassifier(estimators=models, voting='hard')
# 	return ensemble

# def evaluate_model(model, X, y):
# 	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# 	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# 	return scores

if __name__ == "__main__":

	print("\n######################################################")
	print("\n\t\tMULTIMODAL FUSION\n")
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


	################## GET DATASETS ##################
	print("-I- Working on video dataset:")
	dataset_video = pd.read_csv("../dataset/dataset_video.csv")
	
	# ## VARPOS
	# dataset = dataset[['date','varpos_a','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	# ## VARPOS-OHTER
	# dataset = dataset[['date','varpos_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	# ## SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','motion_reaction_a','motion_reaction_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	# ## MOTION-SYNC
	dataset_video = dataset_video[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	X_train_video, y_train_video = bs.process_dataset(dataset_video, target_names)

	print("\n-I- Working on wearables dataset:")
	dataset_wearables = pd.read_csv("../dataset/dataset_wearables.csv")
	X_train_wearable, y_train_wearable = bs.process_dataset(dataset_wearables, target_names)

	X_video_sh, y_video_sh, X_wearable_sh, y_wearable_sh = shuffle(X_train_video, y_train_video, X_train_wearable, y_train_wearable)
	
	X_video = np.asarray(X_video_sh)
	X_wearable = np.asarray(X_wearable_sh)

	X = [X_video, X_wearable]
	#################################################

	knn_video = KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=3, p=1, weights='uniform')
	lr_wearable = LogisticRegression(C=0.00001, max_iter=100000, solver='liblinear')
	svm_video = svm.SVC(C=2, kernel='rbf', probability=True)

	majority_vote = ec.majority_vote(clfs=[svm_video, lr_wearable])

	targets = y_wearable_sh.columns.values
	print("\n-I- Get AUC matrics:")
	for target in targets[0:4]:

		print("-I- Working on: ",target,end = '')

		y = [np.asarray(y_video_sh[target]), np.asarray(y_wearable_sh[target])]

		bs.get_roc_curve_multimodal(majority_vote, X, y, target, splits=10)



