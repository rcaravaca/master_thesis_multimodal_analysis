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
import sys, os
from imblearn.under_sampling import RandomUnderSampler

sys.path.append('/home/rcaravaca/Documents/Maestria/Tesis_Msc/master_thesis_multimodal_analysis/baselines/baselines_functions.py')
import baselines_utils as bu
import ensemble_classifier as ec

import warnings
warnings.filterwarnings('ignore')

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
	sex = '_all'
	# dataset = dataset[dataset['date'].str.contains(sex)]
	day = '^13'
	dataset = dataset[dataset['date'].str.contains(day)]
	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)
	

	# exit()
	### VARPOS
	# dataset = dataset[['date','varpos_a','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARPOS-OHTER
	# dataset = dataset[['date','varpos_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARDIS
	# dataset = dataset[['date','vardis','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','motion_reaction_a','motion_reaction_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### MOTION-SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	X_video, Y_video = bu.process_dataset(dataset, target_names)

	###--------
	# dataset = dataset[dataset['date'].str.contains(sex)]
	# dataset = pd.read_csv("../dataset/dataset_video_baseline.csv")
	# day = '^13'
	# dataset = dataset[dataset['date'].str.contains(day)]
	# dataset = dataset.reset_index()
	# dataset = dataset.drop(["index"], axis=1)
	

	# exit()
	### VARPOS
	# dataset = dataset[['date','varpos_a','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARPOS-OHTER
	# dataset = dataset[['date','varpos_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### VARDIS
	# dataset = dataset[['date','vardis','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','motion_reaction_a','motion_reaction_b','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	### MOTION-SYNC
	# dataset = dataset[['date','motionsync_a_1','motionsync_a_2','motionsync_a_3','motionsync_a_4','M_1', 'M_2', 'M_3', 'M_4', 'M_5', 'M_6', 'PP_M_1', 'PP_M_2', 'PP_M_3', 'PP_M_4', 'PP_M_5', 'PP_M_6']]
	# X_video_test, Y_video_test = bu.process_dataset(dataset, target_names)
	#################################################

	################## GET WEARABLE DATASET #########
	print("\n######################################################")
	print("\n\t\tWEARABLE BASELINE\n")
	print("######################################################\n")

	dataset = pd.read_csv("../dataset/dataset_wearables.csv")

	# dataset = dataset[dataset['date'].str.contains(sex)]
	day = '^13'
	dataset = dataset[dataset['date'].str.contains(day)]
	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)

	dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='sym')))]

	X_wearable, Y_wearable = bu.process_dataset(dataset, target_names)

	### -----
	# dataset = pd.read_csv("../dataset/dataset_wearables.csv")
	# day = '^13'
	# dataset = dataset[dataset['date'].str.contains(day)]
	# dataset = dataset.reset_index()
	# dataset = dataset.drop(["index"], axis=1)

	# dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='sym')))]

	# X_wearable_test, Y_wearable_test = bu.process_dataset(dataset, target_names)

	###################### EXPERIMENT ###############


	experiment = "sync_only"+str(sex)

	os.system("rm "+str(experiment) +".csv")
	output_csv = open(experiment +".csv", "a")

	#################################################
	
	# Xv_sh_test, Yv_sh_test, Xw_sh_test, Yw_sh_test = shuffle(X_video_test, Y_video_test, X_wearable_test, Y_wearable_test)
	targets = Y_wearable.columns.values

	# Xv_test = np.asarray(Xv_sh_test)
	# Xw_test = np.asarray(Xw_sh_test)

	#################################################
	knn_video = KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=3, p=1, weights='uniform')
	lr_wearable = LogisticRegression(C=0.00001, max_iter=100000, solver='liblinear')
	# lr_wearable = LogisticRegression(C=0.00001, max_iter=100000)
	svm_video = svm.SVC(C=2, kernel='rbf', probability=True)
	svm_wearable = svm.SVC(C=2, kernel='rbf', probability=True)

	wearable_model = svm_wearable
	# majority_vote = ec.majority_vote(clfs=[svm_video, wearable_model])

	majority_vote2 = bu.majority_vote_class(clfs=[svm_video, wearable_model], weights=[1,1])

	#################################################

	cv = StratifiedKFold(n_splits=10)
	#################################################

	mean_auc_v = std_auc_v = predict_probas_v = decision_function_v = 0
	mean_auc_w = std_auc_w = predict_probas_w = decision_function_w = 0
	mean_auc_mv = std_auc_mv = 0

	print("\n-I- Get AUC matrics:")
	a = ",des_func_v_str"*144
	b = ",des_func_w_str"*144
	c = ",des_func_mv_str"*144
	output_csv.write("Target,Video_AUC,Wearable_AUC,Majority_vote_AUC"+str(a)+str(b)+str(c)+"\n")

	# smote = SMOTE()
	rus = RandomUnderSampler()

	for target in targets:

		# print(target, end="")
		Xv_sh, Yv_sh, Xw_sh, Yw_sh = shuffle(X_video, Y_video[target], X_wearable, Y_wearable[target])

		Xv_a = np.asarray(Xv_sh)
		Xw_a = np.asarray(Xw_sh)

		Yv = np.asarray(Yv_sh)
		Yw = np.asarray(Yw_sh)

		#############################################
		if "MatchNOT" in target: 
			Xv, Yv = rus.fit_resample(Xv_a, Yv)
			id_rus = rus.sample_indices_

			Xw = Xw_a[id_rus]
			Yw = Yw[id_rus]

			Xv, Yv, Xw, Yw = shuffle(Xv, Yv, Xw, Yw)
		else:
			Xv = Xv_a
			Xw = Xw_a
		#############################################

		# Xv, Yv = smote.fit_resample(Xv, Yv)
		# Xw, Yw = smote.fit_resample(Xw, Yw)

		# Yv_test = np.asarray(Yv_sh_test[target])
		# Yw_test = np.asarray(Yw_sh_test[target])		

		# print((Yv == 0).sum())
		# print((Yv == 1).sum())

		# print(Yv)
		# print(Yw)

		splits_train = cv.split(Xw, Yw)


		# splits_test = cv.split(Xv_test, Yv_test)
		
		trains = []
		tests = []
		for i , (train, test) in enumerate(splits_train):
			trains.append(train)
			tests.append(test)

		# for i , (train, test) in enumerate(splits_test):
		# 	tests.append(test)


		mean_auc_v, std_auc_v, predict_probas_v, decision_function_v, conf_matrix_v, predictions_v = bu.get_roc_curve(svm_video, Xv, Yv, target, trains, tests, "video")
		bu.img_confusion_matrix(conf_matrix_v, [0, 1], title="Video classifier for "+str(target)+" target")

		mean_auc_w, std_auc_w, predict_probas_w, decision_function_w, conf_matrix_w, predictions_w = bu.get_roc_curve(wearable_model, Xw, Yw, target, trains, tests, "wearables")
		bu.img_confusion_matrix(conf_matrix_w, [0, 1], title="Wearables classifier for "+str(target)+" target")

		mean_auc_mv, std_auc_mv, prob, decision_function_mv, conf_matrix_mv = bu.get_roc_curve_multimodal(majority_vote2, [Xv, Xw], [Yv, Yw], target, trains, tests, "Majority_vote")
		bu.img_confusion_matrix(conf_matrix_mv, [0, 1], title="Majority vote classifier for "+str(target)+" target")
		# mean_auc_mv, std_auc_mv, prob, decision_function_mv = bu.majority_vote(predict_probas_v, predict_probas_w, decision_function_v, decision_function_w, tests, Yv, target, "majority_vote")

		a = str(',%0.2f+-%0.2f,%0.2f+-%0.2f,%0.2f+-%0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w, mean_auc_mv, std_auc_mv))

		final_des_func_v = np.concatenate( decision_function_v, axis=0 )
		final_des_func_w = np.concatenate( decision_function_w, axis=0 )
		final_des_func_mv = np.concatenate( decision_function_mv, axis=0 )

		final_predictions_v = np.concatenate( predictions_v, axis=0 )
		final_predictions_w = np.concatenate( predictions_w, axis=0 )

		des_final_predictions_v = np.array2string(final_predictions_v, precision=9, separator=',', max_line_width=100000)
		des_final_predictions_w = np.array2string(final_predictions_w, precision=9, separator=',', max_line_width=100000)

		des_func_v_str = np.array2string(final_des_func_v, precision=9, separator=',', max_line_width=100000)
		des_func_w_str = np.array2string(final_des_func_w, precision=9, separator=',', max_line_width=100000)
		des_func_mv_str = np.array2string(final_des_func_mv, precision=9, separator=',', max_line_width=100000)

		print(str(target)+str('\t%0.2f+-%0.2f\t%0.2f+-%0.2f\t%0.2f+-%0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w, mean_auc_mv, std_auc_mv)))
		# print(str(target)+str('\t%0.2f+-%0.2f\t%0.2f+-%0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w)))
		output_csv.write(str(target)+a+","+des_func_v_str[1:-1]+","+des_func_w_str[1:-1]+","+des_func_w_str[1:-1]+"\n")
		# output_csv.write(str(target)+a+","+des_final_predictions_v[1:-1]+","+des_final_predictions_w[1:-1]+"\n")

	output_csv.close()

if __name__ == "__main__":

	baselines_excecution()
