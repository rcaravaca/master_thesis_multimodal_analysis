"""
Created on 08/30/2021

@author: Ronald
"""


import numpy as np
import sys
import cv2 
import os
import glob
import subprocess
import argparse
import logging

def make_log_file(file_name):

	if os.path.exists(file_name):
		msg("Removing existing file: "+str(file_name))
		os.remove(file_name)
		
	# logging.basicConfig(filename=file_name, encoding='utf-8', level=logging.DEBUG)

	#initialize
	level = "info"
	handlers = [logging.StreamHandler(), logging.FileHandler(file_name, mode="w")]
	logging.basicConfig(
		format="%(message)s",
		level=getattr(logging, level.upper()),
		handlers=handlers
	)

def msg(msg, tp=None):

	if (tp == None or tp == "I") and (msg != "" or msg != "\n"):
		logging.info("UTILS_INFO==> " + msg )
	elif tp == "E":
		logging.error("UTILS_ERROR==> " + msg )
	elif tp == "W":
		logging.warning("UTILS_WARNING==> " + msg )
	else:
		logging.info("\n")

def mkdir(directory):
	if not os.path.exists(directory):
		msg("Creating dir: " +str(directory), "I")
		os.makedirs(directory)
	else:
		msg("dir: " +str(directory) + " already exists.", "W")
