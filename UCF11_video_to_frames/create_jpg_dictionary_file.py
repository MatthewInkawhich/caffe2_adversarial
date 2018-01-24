# NAI
# This script walks through the UCF11 jpg directory that was created with mpg_to_jpgs
#   script and creates a dictionary file of ALL jpgs in that directory structure. The
#   dictionary is a list of <path/to/jpg> <integer label> pairs and is saved in txt
#   based format. This full dictionary should be split into train/test/validate parts then
#   can be used to create lmdbs

import os
import glob

# ***************************************************************
# function to read the key file and return a dictionary that maps english categories to int labels
def read_key_file(key_file):
	# this function reads the keyfile and returns a map whose keys are
	#	the categories in english and whose value is the int label
	key_dict = {}

	f = open(key_file,"r")
	for line in f:
		dat = line.split()
		key_dict[dat[0]] = dat[1]
	f.close()

	return key_dict

# ***************************************************************
# MAIN

# path to root of ucf jpg directory
root_ucf_jpg_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11_updated_jpg_5FPS")
# path to key file for ucf11
ucf_key_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/KEYFILE_UCF11.txt")
# name of output file to save dictionary
ofile_name = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/FullDictionary_UCF11_updated_jpg_5FPS.txt")

f = open(ofile_name, 'wb')

# Get the dictionary of key value pairs from the key file
key_dict = read_key_file(ucf_key_file)

# for each subdirectory in root dir [ex. UCF11_updated_mpg/basketball]
for dir1 in glob.glob(root_ucf_jpg_directory + '/*'):

	# for each subdirectory in new category level directory [ex. UCF11_updated_mpg/basketball/v_shooting_01]
	for dir2 in glob.glob(dir1 + "/*"):

		# extract the category name and decide what the integer label will be
		# ex. tmp = v_shooting_01
		tmp = os.path.split(dir2)[-1]

		# split tmp on "_" and extract the second item, which is the category in english
		# ex. category = shooting
		category = tmp.split("_")[1]

		# find the category in the dict and get its label
		# Note: this will error out if category does not exist in the dictionary (shouldnt happen anyway)
		label = key_dict[category]

		# for each level 3 directory [ex. UCF11_updated_mpg/basketball/v_shooting_01/01/]
		for dir3 in glob.glob(dir2 + "/*"):

			# for each .jpg in directory [ex. UCF11_updated_mpg/basketball/v_shooting_01/01/jpgs/v_shooting_01_01_f0.jpg]
			for img in glob.glob(dir3 + "/jpgs/*.jpg"):
			
				# construct the line to be written to the file
				# ex. </full/path/to/jpg> <label>
				output = img + " " + label
				f.write(output + "\n")
				print output

f.close()
