# NAI
# This script converts the .mpg files from the UCF11_updated_mpg dataset to multiple frames,
#   saved as jpgs, sampled at a certain rate. This script creates another (completely new)
#   UCF11 directory with the same structure as the UCF_update_mpg directory but it has jpgs
#   instead of mpgs. This should only be run once to generate all of the jpgs for a certain
#   sampling rate.
#   This script is CUSTOM made for the UCF11 directory structure!!!


import os
import glob
import cv2


# ***************************************************************
# mpg2jpgs
# 
# This function takes a mpg file, samples the video at intervals of sampling_rate
# 	and saves each sample in the directory specified.
# Note: the videos are 30FPS, so the effective FPS is (30/sampling_rate) 
# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

def mpg2jpgs(mpg_file, output_dir, sampling_rate):
    # create a video capture so we can read the frames
    vidcap = cv2.VideoCapture(vid)
    # read once
    success,image = vidcap.read()
    count = 0
    success = True

    # while there are still frames to be read
    while success:

        # read a single frame
        success,image = vidcap.read()
        
        # check with the sampling rate if we are keeping this one
        if (count % sampling_rate) == 0:
            
            # construct output jpg name
            mpg_fname = os.path.split(mpg_file)[-1]
            shortname,_ = os.path.splitext(mpg_fname)
            jpg_name = shortname + "_f" + str(count) + ".jpg"

            # final, full path name to jpg
            ofname = output_dir+'/'+jpg_name

            # make sure image is something (sometimes image can be empty and a bad jpg gets saved)
            if hasattr(image, 'size'):
                print "\t\tjpg_name: ",ofname

                # pull the trigger
                # save frame as JPEG file
                cv2.imwrite(ofname, image)

        count += 1


# ***************************************************************
# MAIN

# root directory for all videos
root_ucf_mpg_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11_updated_mpg")

# Root JPG directory that will be created by this script
# The directory structure will be the same as the orig mpg directory
root_ucf_jpg_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11_updated_jpg_5FPS")

# for each subdirectory in root dir [ex. UCF11_updated_mpg/basketball]
for dir1 in glob.glob(root_ucf_mpg_directory + '/*'):

	print "dir1: ",dir1

	# for each subdirectory in new category level directory [ex. UCF11_updated_mpg/basketball/v_shooting_01]
	for dir2 in glob.glob(dir1 + "/*"):

		# check if this is the annotation dir and skip if it is
		if os.path.split(dir2)[-1] == "Annotation":
			continue;

		print "\tdir2: ",dir2

		# for each .mpg in directory 
		for vid in glob.glob(dir2 + "/*.mpg"):
			
			print "\t\tvid: ",vid

			# extract the fname without the extension
			vid_name = os.path.split(vid)[-1]
			vid_name,_ = os.path.splitext(vid_name)

			# extract the last number in the fname 
			# ex. v_shooting_01_02.mpg ==> extract '02'
			seq_num = vid_name.split("_")[-1]

			odir = dir2.replace(root_ucf_mpg_directory,root_ucf_jpg_directory) + '/' + seq_num + '/jpgs'
			
			print "vid_name = ",vid_name
			print "seq_num = ",seq_num
			print "odir = ",odir

			# construct the directory if it does not exist
			# opencv will not write an image to a dir that does not exist
			if os.path.exists(odir) == False:
				os.makedirs(odir)

			# perform conversion
			# def mpg2jpgs(mpg_file, output_dir, sampling_rate)
			mpg2jpgs(vid, odir, 6)

			#exit()

