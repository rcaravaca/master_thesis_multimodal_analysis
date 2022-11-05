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

def baselines_excecution(sex):

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

	################## GET VIDEO DATASET ############
	dataset = pd.read_csv("../dataset/golden_dataset/dataset_video.csv")

	# sex = '_all'

	if sex != "_all":
		print("Running for ",sex)
		dataset = dataset[dataset['date'].str.contains(sex)]

	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)
	

	X_video, Y_video = bu.process_dataset(dataset, target_names)
	# X_video.to_csv('video_X.csv', index=False)
	# Y_video.to_csv('video_Y.csv', index=False)

	################## GET WEARABLE DATASET #########
	print("\n######################################################")
	print("\n\t\tWEARABLE BASELINE\n")
	print("######################################################\n")

	dataset = pd.read_csv("../dataset/golden_dataset/dataset_wearables.csv")

	if sex != "_all":
		print("Running for ",sex)
		dataset = dataset[dataset['date'].str.contains(sex)]

	dataset = dataset.reset_index()
	dataset = dataset.drop(["index"], axis=1)

	dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='sym')))]
	X_wearable, Y_wearable = bu.process_dataset(dataset, target_names)
	# X_wearable.to_csv('wearables_X.csv', index=False)
	# Y_wearable.to_csv('wearables_Y.csv', index=False)

	# exit()
	###################### EXPERIMENT ###############

	experiment = "sync_only"+str(sex)

	os.system("rm "+str(experiment) +".csv")
	os.system("rm auc_"+str(experiment) +".csv")
	os.system("rm metrics_"+str(experiment) +".csv")
	output_csv = open(experiment +".csv", "a")
	auc_output_csv = open("auc_"+experiment +".csv", "a")
	metrics_output_csv = open("metrics_"+experiment +".csv", "a")

	#################################################
	
	targets = Y_wearable.columns.values

	#################################################
	# knn_video = KNeighborsClassifier(algorithm='auto', leaf_size=2, n_neighbors=3, p=1, weights='uniform')
	# lr_wearable = LogisticRegression(C=0.00001, max_iter=100000, solver='liblinear')
	svm_video = svm.SVC(C=2, kernel='rbf', probability=True)
	svm_wearable = svm.SVC(C=2, kernel='rbf', probability=True)
	wearable_model = svm_wearable
	# majority_vote2 = bu.majority_vote_class(clfs=[svm_video, wearable_model], weights=[1,1])

	#################################################

	mean_auc_v = std_auc_v = predict_probas_v = decision_function_v = 0
	mean_auc_w = std_auc_w = predict_probas_w = decision_function_w = 0
	mean_auc_mv = std_auc_mv = 0

	print("\n-I- Get AUC matrics:")

	# cv = StratifiedKFold(n_splits=10)

	if sex == "_all":
		d1s = 354
		d2s = 172
		d3s = 144
		a = ",des_func_v_str"*670
		b = ",des_func_w_str"*670
		c = ",des_func_mv_str"*670
	else:
		d1s = 177
		d2s = 86
		d3s = 72
		a = ",des_func_v_str"*335
		b = ",des_func_w_str"*335
		c = ",des_func_mv_str"*335

	output_csv.write("Target,Video_AUC,Wearable_AUC,Majority_vote_AUC"+str(a)+str(b)+str(c)+"\n")
	metrics_output_csv.write("Observation,Classifier,Target,accuracy,f1,precision,recall,auc_pr,AP\n")

	d1 = range(0,d1s)
	d2 = range(d1s,d1s+d2s)
	d3 = range(d1s+d2s,d1s+d2s+d3s)

	trains = [list(d2)+list(d3), list(d1)+list(d3), list(d1)+list(d2)]
	tests = [list(d1), list(d2), list(d3)]
	

	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                'gamma': [0.001, 0.01, 0.1, 1],
		        'kernel': ['rbf', 'poly', 'sigmoid'],
		        }

	targets = ["Sexual", "Romantic"]
	for target in targets:

		print("\n----> Working on target: ", target)

		score = "f1"

		grid_v = GridSearchCV(svm.SVC(probability=True, class_weight='balanced'),param_grid,refit=True,verbose=0, scoring=score, cv=3)
		grid_w = GridSearchCV(svm.SVC(probability=True, class_weight='balanced'),param_grid,refit=True,verbose=0, scoring=score, cv=3)

		Xv = np.asarray(X_video)
		Yv = np.asarray(Y_video[target])

		print("Running video tuning_parameters...")

		grid_v.fit(Xv,Yv)
		svm_video = grid_v.best_estimator_

		Xw = np.asarray(X_wearable)
		Yw = np.asarray(Y_wearable[target])

		print("Running wearables tuning_parameters...")

		grid_w.fit(Xw,Yw)
		wearable_model = grid_w.best_estimator_

		majority_vote2 = bu.majority_vote_class(clfs=[svm_video, wearable_model], weights=[1,1])
		
		auc_v_observations = []
		auc_w_observations = []
		auc_mv_observations = []

		conf_matrix_lst_v = []
		conf_matrix_lst_w = []
		conf_matrix_lst_mv = []

		print("Runnuning Observations...")
		observations = 10
		for o in range(observations):

			if o%2 == 0:
				print("Observation: ",o)

			###########################
			auc_v, decision_function_v, conf_matrix_v, accuracy, f1, prec_score, rec_score, auc_precision_recall, mean_ap_pr, std_ap_pr = bu.get_roc_curve(svm_video, Xv, Yv, target, trains, tests, "video"+sex)

			for mtx in conf_matrix_v:
				conf_matrix_lst_v.append(mtx)
			for auc_ in auc_v:
				auc_v_observations.append(auc_)

			metrics_output_csv.write("{},{},{},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4}\n".format(str(o),"video",target,np.mean(accuracy), np.std(accuracy),np.mean(f1), np.std(f1),np.mean(prec_score), np.std(prec_score),np.mean(rec_score), np.std(rec_score),np.mean(auc_precision_recall), np.std(auc_precision_recall), mean_ap_pr, std_ap_pr))

			###########################
			auc_w, decision_function_w, conf_matrix_w, accuracy, f1, prec_score, rec_score, auc_precision_recall, mean_ap_pr, std_ap_pr = bu.get_roc_curve(wearable_model, Xw, Yw, target, trains, tests, "wearables"+sex)
			
			for mtx in conf_matrix_w:
				conf_matrix_lst_w.append(mtx)
			for auc_ in auc_w:
				auc_w_observations.append(auc_)

			metrics_output_csv.write("{},{},{},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4}\n".format(str(o),"wearable",target,np.mean(accuracy), np.std(accuracy),np.mean(f1), np.std(f1),np.mean(prec_score), np.std(prec_score),np.mean(rec_score), np.std(rec_score),np.mean(auc_precision_recall), np.std(auc_precision_recall), mean_ap_pr, std_ap_pr))

			###########################
			auc_mv, decision_function_mv, conf_matrix_mv, accuracy, f1, prec_score, rec_score, auc_precision_recall, mean_ap_pr, std_ap_pr = bu.get_roc_curve_multimodal(majority_vote2, [Xv, Xw], [Yv, Yw], target, trains, tests, "Majority_vote"+sex)
			
			for mtx in conf_matrix_mv:
				conf_matrix_lst_mv.append(mtx)
			for auc_ in auc_mv:
				auc_mv_observations.append(auc_)

			metrics_output_csv.write("{},{},{},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4},{:.4}+-{:.4}\n".format(str(o),"Maj_vote",target,np.mean(accuracy), np.std(accuracy),np.mean(f1), np.std(f1),np.mean(prec_score), np.std(prec_score),np.mean(rec_score), np.std(rec_score),np.mean(auc_precision_recall), np.std(auc_precision_recall), mean_ap_pr, std_ap_pr))

			###########################
			# auc_v_observations.append(mean_auc_v)
			# auc_w_observations.append(mean_auc_w)
			# auc_mv_observations.append(mean_auc_mv)

		# mean_auc_mv, std_auc_mv, prob, decision_function_mv = bu.majority_vote(predict_probas_v, predict_probas_w, decision_function_v, decision_function_w, tests, Yv, target, "majority_vote")
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_v, axis=0), np.std(conf_matrix_lst_v, axis=0), [1, 0], title="Video classifier for "+str(target)+" target")
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_v, axis=0), np.std(conf_matrix_lst_v, axis=0), [1, 0], title="Norm Video classifier for "+str(target)+" target", norm=True)
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_w, axis=0), np.std(conf_matrix_lst_w, axis=0), [1, 0], title="Wearables classifier for "+str(target)+" target")
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_w, axis=0), np.std(conf_matrix_lst_w, axis=0), [1, 0], title="Norm Wearables classifier for "+str(target)+" target", norm=True)
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_mv, axis=0), np.std(conf_matrix_lst_mv, axis=0), [1, 0], title="Majority vote classifier for "+str(target)+" target")
		bu.img_confusion_matrix(np.mean(conf_matrix_lst_mv, axis=0), np.std(conf_matrix_lst_mv, axis=0), [1, 0], title="Norm Majority vote classifier for "+str(target)+" target", norm=True)
		
		a = str(',%0.2f+-%0.2f,%0.2f+-%0.2f,%0.2f+-%0.2f' % (np.mean(auc_v_observations), np.std(auc_v_observations), 
															np.mean(auc_w_observations), np.std(auc_w_observations), 
															np.mean(auc_mv_observations), np.std(auc_mv_observations)))

		final_des_func_v = np.concatenate( decision_function_v, axis=0 )
		final_des_func_w = np.concatenate( decision_function_w, axis=0 )
		final_des_func_mv = np.concatenate( decision_function_mv, axis=0 )

		# final_predictions_v = np.concatenate( predictions_v, axis=0 )
		# final_predictions_w = np.concatenate( predictions_w, axis=0 )

		# des_final_predictions_v = np.array2string(final_predictions_v, precision=9, separator=',', max_line_width=100000)
		# des_final_predictions_w = np.array2string(final_predictions_w, precision=9, separator=',', max_line_width=100000)

		des_func_v_str = np.array2string(final_des_func_v, precision=9, separator=',', max_line_width=100000)
		des_func_w_str = np.array2string(final_des_func_w, precision=9, separator=',', max_line_width=100000)
		des_func_mv_str = np.array2string(final_des_func_mv, precision=9, separator=',', max_line_width=100000)

		auc_v_observations_str = np.array2string(np.asarray(auc_v_observations), precision=9, separator=',', max_line_width=100000)
		auc_w_observations_str = np.array2string(np.asarray(auc_w_observations), precision=9, separator=',', max_line_width=100000)
		auc_mv_observations_str = np.array2string(np.asarray(auc_mv_observations), precision=9, separator=',', max_line_width=100000)

		# print(np.mean(auc_mv_observations), np.std(auc_mv_observations))
		auc_output_csv.write(str(target)+","+auc_v_observations_str[1:-1]+","+auc_w_observations_str[1:-1]+","+auc_mv_observations_str[1:-1]+"\n")
		# print(str(target)+str('\t%0.2f+-%0.2f\t%0.2f+-%0.2f\t%0.2f+-%0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w, mean_auc_mv, std_auc_mv)))
		# print(str(target)+str('\t%0.2f+-%0.2f\t%0.2f+-%0.2f' % (mean_auc_v, std_auc_v, mean_auc_w, std_auc_w)))
		output_csv.write(str(target)+a+","+des_func_v_str[1:-1]+","+des_func_w_str[1:-1]+","+des_func_mv_str[1:-1]+"\n")
		# output_csv.write(str(target)+a+","+des_final_predictions_v[1:-1]+","+des_final_predictions_w[1:-1]+"\n")

		

	output_csv.close()
	auc_output_csv.close()
	metrics_output_csv.close()

if __name__ == "__main__":

	for sex in ["_all"]:
	# for sex in ["_F"]:
		baselines_excecution(sex)
		# os.system("mv *pdf sync_only_* auc_sync_only_* metrics_*csv WCLASS_VIDEO_BL_ONE_DAY_OUT_EXPERIMENTS/sync_only"+sex)

