#!/usr/bin/env python3

import numpy as np
import cv2
from progress.bar import FillingSquaresBar
import argparse
import matplotlib.pyplot as plt
import video_check as vc
from progress.bar import Bar


def optical_flow(video=None, scale=1, blur=True, show_img=False, plot_mag=False):

	oflow_params = dict(
						pyr_scale=0.5,
						levels=5, 
						winsize=10, 
						iterations=1, 
						poly_n=5, 
						poly_sigma=1.1, 
						flags=0
					)

	sigma = 100
	kernel = cv2.getGaussianKernel(340,sigma)
	kernel = kernel.T*kernel


	if show_img:
		cv2.namedWindow('Original')
		cv2.namedWindow('Optical flow')

	if video is not None:

		video = cv2.VideoCapture(video)

		height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

		H_DIM = int(scale*height)
		W_DIM = int(scale*width)
		num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

		# to visualize
		hsv = np.zeros((H_DIM,W_DIM,3), dtype=np.uint8)
		hsv[:,:,1] = 255

		# to save magnitud
		magnitude = []
		angle = []
		count_frames = 0
		

		bar = Bar('Processing optical flow', max=num_frames)
		while(video.isOpened()):

			ret, frame_bgr = video.read()

			if ret:

				frame = cv2.resize(frame_bgr, (H_DIM,W_DIM), interpolation = cv2.INTER_AREA)
				gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
				# gray_frame = cv2.normalize(gray_frame_orig*kernel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

				if blur:
					ksize = (5, 5)
					gray_frame = cv2.medianBlur(gray_frame,9)

				if count_frames > 0:
					
					# get optical flow, (x,y) by pixel displacement
					flow = cv2.calcOpticalFlowFarneback(prev_frame,gray_frame, None, **oflow_params)
					mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
					
					magnitude.append(mag)
					angle.append(ang)

					if show_img:

						hsv[:,:,0] = ang*180/np.pi/2
						hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
						bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

						gray_frame_copy = gray_frame.copy()
						bgr = vc.drawing_grid(bgr, step = 113, mirror=False)
						gray_frame_copy = vc.drawing_grid(gray_frame_copy, step = 113, mirror=False)

						cv2.imshow('Optical flow',bgr)
						cv2.imshow('Original',gray_frame_copy)

						key = cv2.waitKey(1) & 0xFF
						if key == ord("q"):
							break
						elif key == ord('s'):
							cv2.imwrite('opticalfb.png',gray_frame_copy)
							cv2.imwrite('opticalhsv.png',bgr)

				prev_frame = gray_frame
				count_frames += 1

			bar.next()
			if count_frames >= num_frames:
				break

		bar.finish()

		cv2.destroyAllWindows()
		video.release()

	return np.dstack(magnitude), np.dstack(angle)

if __name__ == "__main__":

	# video = "../msc_cleaning_dataset/spliter/individual_videos/day_2/p24/participant_p24_date_9_Speed_date_part_2_cam02_0_8670.avi"
	video = "../msc_cleaning_dataset/spliter/individual_videos/day_2/p4/participant_p4_date_9_Speed_date_part_2_cam02_1_8670_RECOVERED.avi"

	magnitude, angle = optical_flow(video=video, scale=1, blur=True, show_img=True, plot_mag=True)

	print(magnitude.shape)