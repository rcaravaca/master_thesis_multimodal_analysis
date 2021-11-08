#!/usr/bin/env python3

"""
Created on 08/17/2021

@author: Ronald
"""

import cv2
import numpy as np
import datetime
import argparse

def get_parser():
	parser = argparse.ArgumentParser(description="Check video")
	
	parser.add_argument("-i","--input", help="Input video")
	parser.add_argument("-s","--grid_step", help="Grid step", type=int)

	return parser


def drawing_grid(frame, step = 25, mirror=False):

	img = frame.copy()

	line_color = (0, 255, 0)
	thickness = 1
	type_ = cv2.LINE_AA
	font = cv2.FONT_HERSHEY_SIMPLEX

	
	x = step
	y = step

	c = 0
	while x < img.shape[1]:
		cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
		cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)

		x += step
		y += step

	x = 0

	size = img.shape

	if mirror:
		c = 8
	else:
		c = 0

	for y in range(0,int(size[0]/step)):
		for x in range(0,int(size[1]/step)):
			cv2.putText(img, str(c), (x*step,y*step+int(step/2)), font, 1, line_color, 1, cv2.LINE_AA)

			if mirror:
				c -= 1
			else:
				c += 1



	# while y < img.shape[0]:

	# 	cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
	# 	y += step

	return img


if __name__ == "__main__":

	#### Get parse arguments
	args = get_parser().parse_args()

	video = args.input
	grid_step = args.grid_step

	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
	cap = cv2.VideoCapture(video)

	w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	print("Size: "+str(w)+"x"+str(h) )

	seconds = int(frames / fps)
	print("duration in frames:", frames)
	video_time = str(datetime.timedelta(seconds=seconds))
	print("duration in seconds:", seconds)
	print("video time:", video_time)

	# Check if camera opened successfully
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	# Read until video is completed
	while(cap.isOpened()):
		# Capture frame-by-frame

		ret, frame = cap.read()
		frame = drawing_grid(frame, step = grid_step)
		if ret == True:

			# Display the resulting frame
			cv2.imshow('Frame',frame)

			

			# Press Q on keyboard to	exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

		# Break the loop
		else: 
			break

	# When everything done, release the video capture object
	cap.release()

	# Closes all the frames
	cv2.destroyAllWindows()