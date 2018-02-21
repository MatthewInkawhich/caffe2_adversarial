# NAI

# This is for the jester dataset

import os
import glob
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'lib'))
import image_manipulation
import optical_flow


# ***************************************************************
# MAIN

dictionary_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TestDictionary_5class.txt")
jester_root_dir = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1")
oflow_root_dir = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1-oflow")

fin = open(dictionary_file,"rb")

# for each line in the dictionary file
for line in fin:

    # Get the relevant path from the line
    # This is the full path to a directory of jpgs for a single video
    # path = /.../datasets/jester/20bn-jester-v1/8769
    path = line.split()[0]
    print path

    # get an array of jpgs in the directory (these are full paths)
    jpg_arr = glob.glob(path + "/*.jpg")

    # make an array of just the file names in the jpg arr
    # names = ['00001.jpg', '00002.jpg', ...]
    names = [os.path.split(img)[-1] for img in jpg_arr]
    names.sort()
    #print names


    # Remove the odd indexes (downsample by 2)
    # This will calculate optical flow between 0,2 ; 2,4 ; 4,6 ; 6,8 ; ...
    cnt = 0
    new_names = []
    for name in names:
        if cnt % 2 != 0:
            new_names.append(name)
        cnt+=1
    names=new_names


    # create the oflow directory for this scene if it does not exist
    if os.path.exists(path.replace("20bn-jester-v1","20bn-jester-v1-oflow")) == False:
        os.makedirs(path.replace("20bn-jester-v1","20bn-jester-v1-oflow"))

    # select consecutive pairs of frames in names to calculate optical flow between
    for i in range(len(names)-1):

        frame1 = names[i]
        frame2 = names[i+1]

        # get the frame number of the second frame
        fnum2 = os.path.splitext(frame2)[0]

        # construct output file name
        ofname = "oflow_" + os.path.splitext(frame1)[0] + "_" + os.path.splitext(frame2)[0] + "_" + str(i)

        print "Optical Flow( ",frame1,', ',frame2,' ) => ',ofname

        # format the input strings with the correct full paths
        in1 = path + "/" + frame1
        in2 = path + "/" +  frame2
        in3 = path.replace("20bn-jester-v1","20bn-jester-v1-oflow") + "/" + ofname + "_h.jpg"
        in4 = path.replace("20bn-jester-v1","20bn-jester-v1-oflow") + "/" + ofname + "_v.jpg"

        # print "\tIn1: ", in1
        # print "\tIn2: ", in2
        # print "\tIn3: ", in3
        # print "\tIn4: ", in4

        # Calculate the optical flow between the frames
        optical_flow.write_optical_flow(in1, in2, in3, in4)
