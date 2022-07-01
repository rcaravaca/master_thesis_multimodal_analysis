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

def get_clusters(inputs, part="PART"):

	matrix = inputs.copy()
	if part == "PART":
		clus_0 = matrix[0,0,:]
		clus_1 = matrix[0,1,:]
		clus_2 = matrix[0,2,:]
		clus_3 = matrix[1,0,:]
		clus_4 = matrix[1,1,:]
		clus_5 = matrix[1,2,:]
		clus_6 = matrix[2,0,:]
		clus_7 = matrix[2,1,:]
		clus_8 = matrix[2,2,:]
	elif part == "PP":
		clus_2 = matrix[0,0,:]
		clus_1 = matrix[0,1,:]
		clus_0 = matrix[0,2,:]
		clus_5 = matrix[1,0,:]
		clus_4 = matrix[1,1,:]
		clus_3 = matrix[1,2,:]
		clus_8 = matrix[2,0,:]
		clus_7 = matrix[2,1,:]
		clus_6 = matrix[2,2,:]
	else:
		clus_2 = None
		clus_1 = None
		clus_0 = None
		clus_5 = None
		clus_4 = None
		clus_3 = None
		clus_8 = None
		clus_7 = None
		clus_6 = None

	return clus_0, clus_1, clus_2, clus_3, clus_4, clus_5, clus_6, clus_7, clus_8

def normalize(inputs, tp=None):

	signal = np.asarray(inputs.copy())

	if tp == None:
		
		signal = signal.reshape(-1,1)
		return normalizing.fit_transform(signal).reshape(1,-1)[0]

	elif tp == "zscore":

		return stats.zscore(signal)

	else:

		utils.msg("Normalizing type should be None or zscore.", "W")
		return None

def ploting_cells_signals(clusters):

	fig, axs = plt.subplots(3, 3)

	c = 0
	for row in range(3):
		for col in range(3):
			# clusters[c][clusters[c] == np.inf] = np.median(clusters[c])
			axs[row, col].plot(clusters[c])
			axs[row, col].grid(True)
			axs[row, col].set_title(c)
			c+=1

	fig.tight_layout()



	### Normalizing angles
	#	
	#	Dir 	Deg	|	Val	|	Norm
	#	-----	----	----	-----
	#	Front 	0	|	1	|	0
	#	Right 	90	|	2 	|	0.3333
	#	behind 	180	|	3 	|	0.6666
	#	left	270	| 	4 	|	1
	#
	# print()
	# print("mean: ",np.mean(normalizing.fit_transform(clus_0[:30].reshape(-1,1))))
	# print("median: ",np.median(normalizing.fit_transform(clus_0[:30].reshape(-1,1))))
	# print("Mode: ",mode(normalizing.fit_transform(clus_0[:30].reshape(-1,1))[:,0]))

	# values, counts = np.unique(normalizing.fit_transform(clus_0[:30].reshape(-1,1))[:,0], return_counts=True)
	# ind = np.argmax(normalizing.fit_transform(clus_0[:30].reshape(-1,1))[:,0])
	# print(values)
	# print(counts)

def parsing_low_level_features(clusters):

	clusters_out = []
	for cluster in clusters:
		cluster[cluster == np.inf] = 0
		clusters_out.append(kf.get_low_level_features(normalize(cluster), 10, fs=30))

	return clusters_out


def plot_signals(mag_cluster_norm_part, ang_cluster_norm_part, plot=True):

	if plot:
		fig, axs = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(8,6), dpi=300)


	frame_count = len(mag_cluster_norm_part[0])
	clusters_count = len(mag_cluster_norm_part)

	seconds = np.linspace(0,frame_count//30+1,frame_count)
	counter = 0
	r = 0
	c = 0
	for cluster in mag_cluster_norm_part:

		signal_mag = cluster
		if plot:
			axs[r, c].plot(seconds,signal_mag)
			axs[r, c].set_title(counter)
			axs[r, c].grid(True)

		counter += 1
		c+=1
		if counter % 3 == 0:
			r+=1
			c=0

	if plot:
		fig.supxlabel('Seconds')
		fig.supylabel('Magnitude')
		plt.tight_layout()
		plt.savefig("magnitud_by_cell.pdf")
		plt.close()

	if plot:
		fig, axs = plt.subplots(3, 3, sharex='all', sharey='all', figsize=(8,6), dpi=300)

	counter = 0
	r = 0
	c = 0
	second_div = 10

	ang_cluster_norm_part_2 = []
	for cluster in ang_cluster_norm_part:

		ang_cluster_norm_part_lit = []
		for i in range(frame_count//second_div + 1):
			# ang_cluster_norm_part_lit.append(np.bincount(np.asarray(ang_cluster_norm_part[r,c,i:i+30])))
			values, counts = np.unique(cluster[i:i+second_div], return_counts=True)
			ind = np.argmax(counts)
			for f in range(second_div):
				ang_cluster_norm_part_lit.append(values[ind])  # prints the most frequent element

		signal = np.asarray(ang_cluster_norm_part_lit[:-1])
		ang_cluster_norm_part_2.append(signal)
		if plot:
			axs[r, c].plot(seconds,signal)
			axs[r, c].set_title(counter)
			axs[r, c].grid(True)
		
		counter += 1
		c+=1
		if counter % 3 == 0:
			r+=1
			c=0

	if plot:
		fig.supxlabel('Seconds')
		fig.supylabel('Direction')
		plt.tight_layout()
		plt.savefig("angles_by_cell.pdf")
		plt.close()


	return mag_cluster_norm_part, ang_cluster_norm_part_2

#### MAIN #########################################
if __name__ == "__main__":

	start_now = datetime.now()
	now = start_now.strftime("%d_%m_%Y__%H_%M_%S")

	log_file = "synchrony_features."+now+".log"
	utils.make_log_file(log_file)

	utils.msg("")
	utils.msg("Extrating synchrony features\n")

	dataset_dir = os.path.dirname("../dataset_version_2/json_version_2/")
	dataset_file = open("video_feauters." + now +".csv", "a")
	json_dates = os.listdir(dataset_dir)

	normalizing = MinMaxScaler()
	precision = 6

	utils.msg("Making headers...\n")
	headers = []
	low_features_types = ["person_corr", "mutual_info", "mimicry_feat", "lagged_correlation"]
	for cell_part in range(9):
		for cell_pp in range(9):
			for lf in low_features_types:
				if lf == "mimicry_feat":
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_0")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_1")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_2")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_3")

					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_0")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_1")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_2")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_3")
				else:
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_mean")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_std")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds0")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds1")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds2")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds3")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds4")
					headers.append("mag_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds5")

					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_mean")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_std")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds0")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds1")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds2")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds3")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds4")
					headers.append("ang_cpart_"+str(cell_part) + "_cpp_" + str(cell_pp) + "_" + str(lf) + "_pds5")

	dataset_file.write("date,"+",".join(headers)+",M_1,M_2,M_3,M_4,M_5,M_6,PP_M_1,PP_M_2,PP_M_3,PP_M_4,PP_M_5,PP_M_6\n")

	utils.msg("Starting to get features...")

	total_dates = len(json_dates)

	for date in json_dates:
		utils.msg("Working on json file: " + str(date) + "... "+ "remaning files: "+str(total_dates))
		total_dates -= 1
		date_path = os.path.join(dataset_dir, date)

		### read json date
		date_dict = read_json(date_path)

		if date_dict is not None:

			################################################
			### Date info
			Date_ID = date_dict.get("Date_ID")
			Event = date_dict.get("Event")
			Date_Order = date_dict.get("Date_Order")
			
			### Participant Info
			Part_ID = date_dict.get("Part_ID")
			Sex = date_dict.get("Sex")
			Part_gnd_truth = date_dict.get("Part_gnd_truth")
			Part_ID_mag = str(Part_ID)+"_mag"
			Part_ID_ang = str(Part_ID)+"_ang"
			
			### Partner Info
			PP_ID = date_dict.get("PP_ID")
			PP_Sex = date_dict.get("PP_Sex")
			PP_gnd_truth = date_dict.get("PP_gnd_truth")
			PP_ID_mag = str(PP_ID)+"_mag"
			PP_ID_ang = str(PP_ID)+"_ang"		


			################################################

			### Magnitud full array ########################
			part_magnitud = np.asarray(date_dict.get(Part_ID_mag))
			pp_magnitud = np.asarray(date_dict.get(PP_ID_mag))

			### Get clusters to list
			mag_clusters_part_aux = get_clusters(part_magnitud, part="PART")
			mag_clusters_pp_aux = get_clusters(pp_magnitud, part="PP")
			
			### Angle full array ###########################
			part_angle = np.asarray(date_dict.get(Part_ID_ang))
			pp_angle = np.asarray(date_dict.get(PP_ID_ang))

			### Get clusters to list
			ang_clusters_part_aux = get_clusters(part_angle, part="PART")
			ang_clusters_pp_aux = get_clusters(pp_angle, part="PP")

			mag_clusters_part, ang_clusters_part = plot_signals(mag_clusters_part_aux, ang_clusters_part_aux, False)
			mag_clusters_pp, ang_clusters_pp = plot_signals(mag_clusters_pp_aux, ang_clusters_pp_aux, False)

			# exit()

			### Parsing low level features
			low_level_mag_clusters_part = parsing_low_level_features(mag_clusters_part)
			low_level_mag_clusters_pp = parsing_low_level_features(mag_clusters_pp)
			low_level_ang_clusters_part = parsing_low_level_features(ang_clusters_part)
			low_level_ang_clusters_pp = parsing_low_level_features(ang_clusters_pp)

			dataset_file.write(Date_ID + "_" + Sex + ",")
			for cell_part in range(9):
				for cell_pp in range(9):
					### Parsing complex features
					### [person_corr (8), mutual_info (8), mimicry_feat (4), lagged_correlation (8)]
					sync_features_mag = kf.get_complex_features(mag_clusters_part[cell_part],mag_clusters_pp[cell_pp],low_level_mag_clusters_part[cell_part],low_level_mag_clusters_pp[cell_pp], True)

					sync_features_ang = kf.get_complex_features(ang_clusters_part[cell_part],ang_clusters_pp[cell_pp],low_level_ang_clusters_part[cell_part],low_level_ang_clusters_pp[cell_pp], True)
			
					
					for feature in sync_features_mag:
						feature_str = np.array2string(np.asarray(feature), precision=precision, separator=',', max_line_width=10000)
						dataset_file.write(feature_str[1:-1]+",")

					for feature in sync_features_ang:
						feature_str = np.array2string(np.asarray(feature), precision=precision, separator=',', max_line_width=10000)
						dataset_file.write(feature_str[1:-1]+",")

			Part_gnd_truth = np.array2string(np.array(Part_gnd_truth,dtype=int), precision=2, separator=',', max_line_width=1000)
			dataset_file.write(Part_gnd_truth[1:-1]+",")
			PP_gnd_truth = np.array2string(np.array(PP_gnd_truth,dtype=int), precision=2, separator=',', max_line_width=1000)
			dataset_file.write(PP_gnd_truth[1:-1]+"\n")

			dataset_file.write(Date_ID + "_" + PP_Sex + ",")
			for cell_pp in range(9):
				for cell_part in range(9):
					### Parsing complex features
					### [person_corr (8), mutual_info (8), mimicry_feat (4), lagged_correlation (8)]
					sync_features_mag = kf.get_complex_features(mag_clusters_pp[cell_pp],mag_clusters_part[cell_part],low_level_mag_clusters_pp[cell_pp],low_level_mag_clusters_part[cell_part], True)

					sync_features_ang = kf.get_complex_features(ang_clusters_pp[cell_pp],ang_clusters_part[cell_part],low_level_ang_clusters_pp[cell_pp],low_level_ang_clusters_part[cell_part], True)

					for feature in sync_features_mag:
						feature_str = np.array2string(np.asarray(feature), precision=precision, separator=',', max_line_width=10000)
						dataset_file.write(feature_str[1:-1]+",")

					for feature in sync_features_ang:
						feature_str = np.array2string(np.asarray(feature), precision=precision, separator=',', max_line_width=10000)
						dataset_file.write(feature_str[1:-1]+",")

			dataset_file.write(PP_gnd_truth[1:-1]+",")
			dataset_file.write(Part_gnd_truth[1:-1]+"\n")
		else:
			utils.msg("Error detected reading json file.\n", "W")


	utils.msg("Dataset was save to: video_feauters." + now +".csv")
	dataset_file.close()

	utils.msg("Check log file: "+str(log_file))
	final_time = datetime.now()
	utils.msg("Elapsed time for all dates: "+str(final_time - start_now))
	
