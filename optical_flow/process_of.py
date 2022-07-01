#!/usr/bin/env python3

"""
Created on 08/17/2021

@author: Ronald
"""


import numpy as np
import sys
import cv2 
import os
import glob
# import subprocess
import argparse
sys.path.append('../miscellaneous/')
import utils
# from natsort import natsorted
import json, codecs
# import pickle
import csv
import optical_flow as of
from datetime import datetime, timedelta
import gc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

### parse inputs arguments
def get_parser():
	parser = argparse.ArgumentParser(description="This script is wrapper to get dense trajectories from ./dense_trajectories/release/DenseTrack")
	
	parser.add_argument("-i","--input_dir", help="Directory where video secuence are")
	parser.add_argument("-o","--output_dir", help="Directory to save the resutls")
	parser.add_argument("-d","--day", help="Day to process", type=int)

	return parser

### Get dense trajectories from csv file in frame
def get_dt_from_csv(csv_file):


	try:
		f=open(csv_file,"r")

		lines=f.readlines()
		list_matrix = []

		for i in lines:
			list_matrix.append(i.split(sep='\t')[:-1])

		f.close()

		Matrix = np.array(list_matrix).astype(np.float)

		return Matrix
	except:
		utils.msg("Can't access csv file...", "E")
		return 0


### get mean of trajectories by clusters
def get_mean_of_trajectories(windows_trajectories):

	windows = windows_trajectories.keys()
	utils.msg("Count of keys of windows_trajectories: "+str(len(windows)))

	# Get means for all trajectorires for each window
	for window in windows:

		utils.msg("***********************************************************")
		utils.msg("For window: "+str(window))
		utils.msg("***********************************************************")

		clusters = windows_trajectories.get(window)

		for cluster in clusters.keys():

			stack_mtx = np.asarray(clusters.get(cluster))
			utils.msg("In cluster: "+str(cluster)+" there are "+str(stack_mtx.shape)+" DT.")

			if stack_mtx.shape[0] > 0:

				mean_of_mtx = stack_mtx.mean(axis=0)
				clusters.update({cluster: mean_of_mtx.tolist()})

				aux_mtx = np.asarray(clusters.get(cluster))
				utils.msg("After get mean there are "+str(aux_mtx.shape)+" DT.")

			elif stack_mtx.shape[0] == 0:

				mean_of_mtx = np.zeros(528,dtype=int)
				clusters.update({cluster: mean_of_mtx.tolist()})
				utils.msg("Set zeros array as mean of DT.")



### save to a json file
def save_json(file, dictionary):

	if os.path.exists(file):
		os.remove(file)

	with open(file, "w") as outfile:
		json.dump(dictionary, outfile)


### read from a json file
def read_json(file):

	if os.path.exists(file):
		obj_text = codecs.open(file, 'r', encoding = 'utf-8').read()
		dictionary = json.loads(obj_text)
		return dictionary
	else:
		utils.msg("File: "+str(file)+" doest not exist!")


front = 1
right = 2
left = 3
behind = 4
def get_angle(A, part="PART"):

	hist, bin_edges = np.histogram(A, bins=np.arange(0,361,45))

	angle = bin_edges[np.where(hist == hist.max())][0]

	if part == "PART":
		if angle == 0 or angle == 315: # front
			orientation = front
		elif angle == 45 or angle == 90: # right
			orientation = right
		elif angle == 135 or angle == 180: # behind
			orientation = behind
		elif angle == 225 or angle == 270: # left
			orientation = left
		else:
			orientation = None
	elif part == "PP":
		if angle == 0 or angle == 315: # behind
			orientation = behind
		elif angle == 45 or angle == 90: # left
			orientation = left
		elif angle == 135 or angle == 180: # front
			orientation = front
		elif angle == 225 or angle == 270: # right
			orientation = right
		else:
			orientation = None
	else:
		orientation = None

	return orientation

### Get optical flow for given video
def get_optical_flow(video_part, part="PART"):

	if os.path.exists(video_part):

		utils.msg(">>>>>>>>>>>>>>>>>>>>>>>>>>>")
		utils.msg("Get optical flow for video: "+str(video_part))
		utils.msg(">>>>>>>>>>>>>>>>>>>>>>>>>>>")

		of_start_time = datetime.now()
		magnitude, angle  = of.optical_flow(video=video_part, scale=1, blur=False, show_img=True, plot_mag=True)
		of_final_time = datetime.now()
		utils.msg("Optical flow elapsed time: "+str(of_final_time - of_start_time))

		mag_cluster = [0,0,0,0,0,0,0,0,0]
		ang_cluster = [0,0,0,0,0,0,0,0,0]

		### Clustering trajestories in a grid of 3x3
		#
		#	0 	114	 228  340
		#	  --- --- ---
		#	 | 0 | 1 | 2 |
		# 114 --- --- ---
		#	 | 3 | 4 | 5 |
		# 228 --- --- ---
		#	 | 6 | 7 | 8 |
		# 340 --- --- ---
		#
		########################################################
		if part == "PART":
		
			mag_cluster[0] = magnitude[:114,:114,:]
			mag_cluster[1] = magnitude[114:228,:114,:]
			mag_cluster[2] = magnitude[228:,:114,:]
			mag_cluster[3] = magnitude[:114,114:228,:]
			mag_cluster[4] = magnitude[114:228,114:228,:]
			mag_cluster[5] = magnitude[228:,114:228,:]
			mag_cluster[6] = magnitude[:114,228:,:]
			mag_cluster[7] = magnitude[114:228,228:,:]
			mag_cluster[8] = magnitude[228:,228:,:]

			ang_cluster[0] = angle[:114,:114,:]
			ang_cluster[1] = angle[114:228,:114,:]
			ang_cluster[2] = angle[228:,:114,:]
			ang_cluster[3] = angle[:114,114:228,:]
			ang_cluster[4] = angle[114:228,114:228,:]
			ang_cluster[5] = angle[228:,114:228,:]
			ang_cluster[6] = angle[:114,228:,:]
			ang_cluster[7] = angle[114:228,228:,:]
			ang_cluster[8] = angle[228:,228:,:]
		
		### Clustering trajestories in a grid of 3x3
		#
		#	0 	114	 228  340
		#	  --- --- ---
		#	 | 8 | 7 | 6 |
		# 114 --- --- ---
		#	 | 5 | 4 | 3 |
		# 228 --- --- ---
		#	 | 2 | 1 | 0 |
		# 340 --- --- ---
		#
		########################################################		
		elif part == "PP":

			mag_cluster[8] = magnitude[:114,:114,:]
			mag_cluster[7] = magnitude[114:228,:114,:]
			mag_cluster[6] = magnitude[228:,:114,:]
			mag_cluster[5] = magnitude[:114,114:228,:]
			mag_cluster[4] = magnitude[114:228,114:228,:]
			mag_cluster[3] = magnitude[228:,114:228,:]
			mag_cluster[2] = magnitude[:114,228:,:]
			mag_cluster[1] = magnitude[114:228,228:,:]
			mag_cluster[0] = magnitude[228:,228:,:]

			ang_cluster[8] = angle[:114,:114,:]
			ang_cluster[7] = angle[114:228,:114,:]
			ang_cluster[6] = angle[228:,:114,:]
			ang_cluster[5] = angle[:114,114:228,:]
			ang_cluster[4] = angle[114:228,114:228,:]
			ang_cluster[3] = angle[228:,114:228,:]
			ang_cluster[2] = angle[:114,228:,:]
			ang_cluster[1] = angle[114:228,228:,:]
			ang_cluster[0] = angle[228:,228:,:]
	
		mag_cluster_norm = np.zeros((3,3,magnitude.shape[2]))
		ang_cluster_norm = np.zeros((3,3,magnitude.shape[2]))

		rows = 0
		cols = 0

		# for each cluster
		for cluster in range(len(mag_cluster)):

			if cluster == 3 or cluster == 6:
				rows += 1
				cols = 0

			# for each frame in cluster
			for frame in range(mag_cluster[cluster].shape[2]):
				mag_cluster_norm[rows,cols,frame] = np.linalg.norm(mag_cluster[cluster][:,:,frame], 'fro')
				ang_cluster_norm[rows,cols,frame] = get_angle(ang_cluster[cluster][:,:,frame], part=part)

			cols += 1

	else:
		utils.msg("File: "+str(video_part)+" does not exist!", "W")

	return mag_cluster_norm, ang_cluster_norm
	# , magnitude, angle

def normalizing(signal):

	signal[signal == np.inf] = 0
	max_val = np.max(signal)
	return signal/max_val


def print_signals(mag_cluster_norm_part, ang_cluster_norm_part):

	fig, axs = plt.subplots(mag_cluster_norm_part.shape[0], mag_cluster_norm_part.shape[0], sharex='all', sharey='all', figsize=(8,6), dpi=700)


	frame_count = mag_cluster_norm_part[0,0,:].shape[0]
	seconds = np.linspace(0,frame_count//30+1,frame_count)
	counter = 0
	for r in range(mag_cluster_norm_part.shape[0]):
		for c in range(mag_cluster_norm_part.shape[0]):

			signal = normalizing(mag_cluster_norm_part[r,c,:])
			axs[r, c].plot(seconds,signal)
			axs[r, c].set_title(counter)
			axs[r, c].grid(True)
			counter += 1

	fig.supxlabel('Seconds')
	fig.supylabel('Magnitude')
	plt.tight_layout()
	plt.savefig("magnitud_by_cell.pdf")
	plt.close()



	fig, axs = plt.subplots(ang_cluster_norm_part.shape[0], ang_cluster_norm_part.shape[0], sharex='all', sharey='all', figsize=(8,6), dpi=700)
	frame_count = ang_cluster_norm_part[0,0,:].shape[0]
	seconds = np.linspace(0,frame_count//30+1,frame_count)
	counter = 0

	
	for r in range(ang_cluster_norm_part.shape[0]):
		for c in range(ang_cluster_norm_part.shape[0]):

			ang_cluster_norm_part_lit = []
			for i in range(frame_count//15 + 1):
				# ang_cluster_norm_part_lit.append(np.bincount(np.asarray(ang_cluster_norm_part[r,c,i:i+30])))
				values, counts = np.unique(ang_cluster_norm_part[r,c,i:i+15], return_counts=True)
				ind = np.argmax(counts)
				for f in range(15):
					ang_cluster_norm_part_lit.append(values[ind])  # prints the most frequent element

			signal = normalizing(ang_cluster_norm_part_lit[:-1])
			axs[r, c].plot(seconds,signal)
			axs[r, c].set_title(counter)
			axs[r, c].grid(True)
			counter += 1

	fig.supxlabel('Seconds')
	fig.supylabel('Direction')
	plt.tight_layout()
	plt.savefig("angles_by_cell.pdf")
	plt.close()

#### MAIN #########################################
if __name__ == "__main__":

	# current date and time
	start_now = datetime.now()
	start_time = start_now.strftime("%d_%m_%Y__%H_%M_%S")

	#### Get parse arguments
	args = get_parser().parse_args()
	d = args.day

	#### LOG FILE
	log_file = "process_of."+start_time+"_day_"+str(d)+".log"
	utils.make_log_file(log_file)

	input_dir = args.input_dir
	days = os.listdir(input_dir)

	output_dir = args.output_dir
	utils.mkdir(output_dir)

	with open('../pairwise_date_response.csv', 'r') as file:

		reader = csv.reader(file)

		for date in reader:

			Event = date[3]
			if date[0] != "Date_ID" and Event == str(d):
			
				Date_ID = date[0]
				Date_Order = date[5]

				ID = date[1]
				Part_Num = date[2]
				Part_gnd_truth = date[6:12]
				Sex = date[12]

				PP_ID = date[13]
				PP_Num = str(int(PP_ID[2:]))
				PP_gnd_truth = date[14:20]
				PP_Sex = date[20]

				day = "day_"+Event
				
				Part_video_name = "participant_p" +Part_Num+ "_date_" +Date_Order+ "_Speed_date_part*.avi"
				PP_video_name = "participant_p" +PP_Num+ "_date_" +Date_Order+ "_Speed_date_part*.avi"

				json_file_output = os.path.join(output_dir, Date_ID+".json")

				if os.path.exists(json_file_output):

					utils.msg("")
					utils.msg("###########################################################")
					utils.msg("json file for date: "+Date_ID + " already exists!")
					utils.msg("###########################################################\n")

				else:

					utils.msg("")
					utils.msg("###########################################################")
					utils.msg("Working on date: "+Date_ID)
					utils.msg("###########################################################\n")

					ini_time_for_video = datetime.now()

					Part_video_path = os.path.join(input_dir, day,"p"+Part_Num,Part_video_name)
					PP_video_path = os.path.join(input_dir, day,"p"+PP_Num,PP_video_name)

					PP_video_path_glob = glob.glob(PP_video_path)
					Part_video_path_glob = glob.glob(Part_video_path)

					if len(Part_video_path_glob) == 0 or len(PP_video_path_glob) == 0:
						utils.msg("One of the participants doesnt have video... json file wont be created for date.\n", "W")

					elif len(Part_video_path_glob) == 1 and len(PP_video_path_glob) == 1:
						utils.msg("Part ID:\t\t"+ID)
						utils.msg("PP ID:\t\t"+PP_ID)
						utils.msg("Event:\t\t"+Event)
						utils.msg("Date_Order:\t"+Date_Order)
						utils.msg("Part path:\t"+Part_video_path_glob[0])
						utils.msg("PP path:\t\t"+PP_video_path_glob[0])

						### get_dense_trajectories(video, out_dir, L) to Part
						# mag_cluster_norm_part, ang_cluster_norm_part = get_optical_flow(glob.glob(Part_video_path)[0], "PART")
						# np.save('mag_cluster_norm_part.npy', mag_cluster_norm_part)
						# np.save('ang_cluster_norm_part.npy', ang_cluster_norm_part)
						mag_cluster_norm_part = np.load("mag_cluster_norm_part.npy")
						ang_cluster_norm_part = np.load("ang_cluster_norm_part.npy")
						print_signals(mag_cluster_norm_part, ang_cluster_norm_part)
						exit()
						### get_dense_trajectories(video, out_dir, L) to PP
						mag_cluster_norm_pp, ang_cluster_norm_pp = get_optical_flow(glob.glob(PP_video_path)[0], "PP")

						temp_dict = {"Date_ID": Date_ID,
									"Part_ID": ID,
									"Sex": Sex,
									"Part_gnd_truth": Part_gnd_truth,
									"PP_ID": PP_ID,
									"PP_Sex": PP_Sex,
									"PP_gnd_truth": PP_gnd_truth,
									"Event": Event,
									"Date_Order": Date_Order,
									"Part_path": Part_video_path_glob[0],
									"PP_path": PP_video_path_glob[0],

									str(ID)+"_mag": mag_cluster_norm_part.tolist(),
									str(ID)+"_ang": ang_cluster_norm_part.tolist(),
									# str(ID)+"_full_mag": magnitude_part.tolist(),
									# str(ID)+"_full_ang": angle_part.tolist(),

									str(PP_ID)+"_mag": mag_cluster_norm_pp.tolist(),
									str(PP_ID)+"_ang": ang_cluster_norm_pp.tolist()
									# str(PP_ID)+"_full_mag": magnitude_pp.tolist(),
									# str(PP_ID)+"_full_ang": angle_pp.tolist()

									}
		
						save_json(json_file_output, temp_dict)
						video_final_time = datetime.now()
						utils.msg("Elapsed time for date "+Date_ID+": "+str(video_final_time - ini_time_for_video))

						temp_dict = {}
						mag_cluster_norm_part = 0
						ang_cluster_norm_part = 0
						# magnitude_part = 0
						# angle_part = 0

						mag_cluster_norm_pp = 0
						ang_cluster_norm_pp = 0
						# magnitude_pp = 0
						# angle_pp = 0
						gc.collect()

					else:
						utils.msg("Multiple paths detected for one of videos!\n", "W")

	utils.msg("Check log file: "+str(log_file))
	final_time = datetime.now()
	utils.msg("Elapsed time for all dates: "+str(final_time - start_now))

