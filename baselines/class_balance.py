#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
from sklearn import preprocessing, svm
from scipy import stats
import sys, os
from imblearn.under_sampling import RandomUnderSampler

sys.path.append('/home/rcaravaca/Documents/Maestria/Tesis_Msc/master_thesis_multimodal_analysis/baselines/baselines_functions.py')
import baselines_utils as bu
import ensemble_classifier as ec

import warnings
warnings.filterwarnings('ignore')


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

c = 0
fig, ax = plt.subplots(1,3)
for sex in ["_all", "_M", "_F"]:
# for sex in ["_all"]:
	
	dataset1 = dataset
	if sex != "_all":
		print("Running for ",sex)
		dataset1 = dataset[dataset['date'].str.contains(sex)]
	else:
		dataset1 = dataset

	dataset1 = dataset1.reset_index()
	dataset1 = dataset1.drop(["index"], axis=1)

	X_video, Y_video = bu.process_dataset(dataset1, target_names)

	df = Y_video
	str(sex)

	zeros = []
	ones = []
	z_per = []
	o_per = []

	for target in df.columns:
		print(target)
		bin_arr = np.bincount(df[target])
		# zeros.append(bin_arr[0])
		# ones.append(bin_arr[1])
		zeros.append(100*(bin_arr[0]/ (bin_arr[0]+bin_arr[1])))
		ones.append(100*(bin_arr[1]/ (bin_arr[0]+bin_arr[1])))

	x_axis = np.arange(len(df.columns))
	bars = ax[c].bar(x_axis -0.2, ones, width=0.4, label = 'Positive')

	for bars in ax[c].containers:
		ax[c].bar_label(bars, fmt="%.1f")

	ax[c].bar(x_axis +0.15, zeros, width=0.3, label = 'Negative')

	ax[c].legend()

	if c == 0:
		ax[c].set_ylabel("Percentage (%)")
		ax[c].set_title("Males and females")
	if c == 1:
		ax[c].set_title("Males")
	if c == 2:
		ax[c].set_title("Females")
	ax[c].set_xticks(x_axis, df.columns, rotation = 75)
	# plt.ylabel("Percentage (%)")
	
	# plt.savefig("balance_plotting"+tag+".pdf")
	c+=1


plt.tight_layout()
plt.show()
