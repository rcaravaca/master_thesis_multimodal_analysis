#!/usr/bin/env python3

"""
Created on 11/22/2021

@author = Ronald
"""

import numpy as np
import sys
import cv2 
import os
import glob
import argparse

import json, codecs
import csv
from datetime import datetime, timedelta
import gc
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from statistics import mode
import kapcak_features as kf

sys.path.append('../miscellaneous/')
import utils


### read from a json file
def read_json(file):

	if os.path.exists(file):
		obj_text = codecs.open(file, 'r', encoding = 'utf-8').read()
		dictionary = json.loads(obj_text)
	else:
		utils.msg("File '"+str(file)+"' doest not exist!", "E")
		dictionary = None

	return dictionary


if __name__ == "__main__":


	dataset_dir = os.path.dirname("../dataset/json_version_2/")

	dataset_file = open("video_row_data.csv", "a")

	json_dates = os.listdir(dataset_dir)


	dataset_file.write("date" + ",")
	features_numbers = list(np.arange(161964))
	features_numbers = ','.join(map(str, features_numbers))
	# features_numbers = np.array2string(, precision=1, separator=',', max_line_width=10000)
	dataset_file.write(features_numbers+",M_1,M_2,M_3,M_4,M_5,M_6,PP_M_1,PP_M_2,PP_M_3,PP_M_4,PP_M_5,PP_M_6\n")

	for json_file in json_dates:

		json_date = read_json(os.path.join(dataset_dir,json_file))

		if json_date is not None:

			print("Working on file: "+str(json_file))

			part_mag_str = np.asarray(json_date.get(str(json_date.get('Part_ID'))+"_mag"))
			part_ang_str = np.asarray(json_date.get(str(json_date.get('Part_ID'))+"_ang"))

			pp_mag_str = np.asarray(json_date.get(str(json_date.get('PP_ID'))+"_mag"))
			pp_ang_str = np.asarray(json_date.get(str(json_date.get('PP_ID'))+"_ang"))

			Part_gnd_truth = json_date.get('Part_gnd_truth')
			PP_gnd_truth = json_date.get('PP_gnd_truth')

			date = json_date.get('Date_ID')
			sex = json_date.get('Sex')
			pp_sex = json_date.get('Sex')

			#########################################################
			final_list = []
			dataset_file.write(date + "_" + sex + ",")
			for i in range(3):
				for j in range(3):
					final_list = final_list + list(part_mag_str[i,j,:])
					final_list = final_list + list(part_ang_str[i,j,:])
					final_list = final_list + list(pp_mag_str[i,j,:])
					final_list = final_list + list(pp_ang_str[i,j,:])
			
			final_list = final_list + list(Part_gnd_truth) + list(PP_gnd_truth)
			final_list = ','.join(map(str, final_list))
			dataset_file.write(final_list + "\n")

			#########################################################
			final_list = []
			dataset_file.write(date + "_" + pp_sex + ",")
			for i in range(3):
				for j in range(3):
					final_list = final_list + list(pp_mag_str[i,j,:])
					final_list = final_list + list(pp_ang_str[i,j,:])
					final_list = final_list + list(part_mag_str[i,j,:])
					final_list = final_list + list(part_ang_str[i,j,:])

			final_list = final_list + list(Part_gnd_truth) + list(PP_gnd_truth)
			final_list = ','.join(map(str, final_list))
			dataset_file.write(final_list + "\n")

			#########################################################
			
		else:
			print("Error detected reading json file.\n")


	dataset_file.close()



	
