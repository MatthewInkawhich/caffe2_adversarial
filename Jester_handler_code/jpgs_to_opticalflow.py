# NAI

# This is for the jester dataset

import os
import glob
import cv2

# ***************************************************************
# Function to calculate dense optical flow between two adjacent frames
def calc_optical_flow(img1, img2, method, ofile_name_horizontal, ofile_name_vertical):
    # global count
    # count += 1
    frame1 = cv2.imread(img1)
    frame2 = cv2.imread(img2)

    f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # calculate dense optical flow
    # settings from tutorial
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

    # NATES CV2
    flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # MATTS CV2
    #flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 15, 3, 5, 1.2, 0)

    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    print "\tBefore adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    if method == 0:


        # From beyond short snippits
        h_oflow[h_oflow < -40] = -40
        h_oflow[h_oflow > 40] = 40
        v_oflow[v_oflow < -40] = -40
        v_oflow[v_oflow > 40] = 40
        
        h_oflow = cv2.normalize(h_oflow, None, 0, 255, cv2.NORM_MINMAX)
        v_oflow = cv2.normalize(v_oflow, None, 0, 255, cv2.NORM_MINMAX)
        
        
        '''
        #print "Method 1: recenter"
        # Recenter the data to 127
        h_oflow += 127
        v_oflow += 127

        h_oflow[h_oflow < 0] = 0
        v_oflow[v_oflow > 255] = 255
        '''

    # if method == 1:
    # 	#print "Method 2: normalize"
    # 	h_oflow = cv2.normalize(h_oflow, None, 0, 255, cv2.NORM_MINMAX)
    # 	v_oflow = cv2.normalize(v_oflow, None, 0, 255, cv2.NORM_MINMAX)
    #else:
    #	print "Abort: unknown method"
    #	exit()

    print "\tAfter adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    # Save the optical flow displacement fields as images
    cv2.imwrite(ofile_name_horizontal, h_oflow)
    cv2.imwrite(ofile_name_vertical, v_oflow)


# ***************************************************************
# MAIN

dictionary_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TrainDictionary_5class.txt")
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

        print "\tIn1: ", in1
        print "\tIn2: ", in2
        print "\tIn3: ", in3
        print "\tIn4: ", in4

        # Calculate the optical flow between the frames
        calc_optical_flow(in1, in2, 0, in3, in4)

