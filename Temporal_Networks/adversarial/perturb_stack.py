##################################################################################
# NAI
#
# This script contains the function used to perturb an optical flow stack,
#	given the gradient of the loss w.r.t. the data. 
#
# First, we use the data_grad to calculate the saliency map of the optical 
#	flow stack.  Then we find the coordinates of the N largest values in 
#	the saliency map. We then perturb each of those locations in the original
#	optical flow stack by the specified technique. The output of the script is
#	the perturbed optical flow stack. This stack can then be fed into a classifier
#	to see if the perturbations caused an adversarial example
##################################################################################

import numpy as np 
import os
import glob
import random
import cv2
import sys
import matplotlib.pyplot as plt
# So we can import jeter dataset handler
sys.path.append("/Users/nathaninkawhich/DukeML/caffe2_sandbox/Temporal_Networks")
import JesterDatasetHandler as jdh 

# Generate inputs
i1_h = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00014_00016_6_h.jpg"
i1_v = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00014_00016_6_v.jpg"

i2_h = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00016_00018_7_h.jpg"
i2_v = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00016_00018_7_v.jpg"

i3_h = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00018_00020_8_h.jpg"
i3_v = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00018_00020_8_v.jpg"

i4_h = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00020_00022_9_h.jpg"
i4_v = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1-oflow/9/oflow_00020_00022_9_v.jpg"

# Generate an optical flow stack
oflow_stack = jdh.create_oflow_stack([i1_h,i1_v,i2_h,i2_v,i3_h,i3_v,i4_h,i4_v])
print "oflow stack shape: ",oflow_stack.shape

# Generate a stand-in for a data_grad 
data_grad = np.random.random(oflow_stack.shape)
print "data grad shape: ",data_grad.shape

# Define the function that will actually perturb the optical flow stack at the most salient points
def perturb_stack(oflow_stack, data_grad, N):

	# Make a copy of the original optical flow stack so we do not overwrite the original
	adv_oflow = np.copy(oflow_stack)

	# Take the element-wise absolute value of the gradient of loss w.r.t data to 
	# 	approximate the saliency map [https://arxiv.org/pdf/1312.6034.pdf]
	saliency_map = np.absolute(data_grad)

	# Find the coordinates of the N largest (i.e. most influential/salient features)
	indices =  np.argpartition(saliency_map.flatten(), -N)[-N:]
	locs =  np.vstack(np.unravel_index(indices, saliency_map.shape)).T

	# Perturb the original optical flow stack at those locations
	for l in locs:

		## Specific perturbation method applied here!!

		# Perturb Method 1: Negate the optical flow stack at each location
		adv_oflow[l[0],l[1],l[2]] *= -1


	# Return the perturbed oflow stack, and the locations of the perturbations
	return adv_oflow, locs








#exit()

##############################################################
# Trying stuff
##############################################################

##############################################################
# How to find coordinates of N largest?
a = np.zeros((4, 4))
a[0,2] = 1
a[1,0] = 9
a[2,2] = 5
#print a

b = np.zeros((4, 4))
b[0,0] = -10
b[0,2] = 10
b[0,3] = 3
b[2,1] = 8
b[2,3] = 6
#print b

c = np.zeros((4, 4))
c[0,3] = 7
c[1,1] = 2
c[3,2] = 10
#print c

stack = np.zeros(shape=(3,4,4))
stack[0,:,:] = a
stack[1,:,:] = b
stack[2,:,:] = c

print stack

# Get the coords of the 3 largest elements of stack
indices =  np.argpartition(stack.flatten(), -3)[-3:]
print np.vstack(np.unravel_index(indices, stack.shape)).T

##############################################################
# Playing with optical flow algorithm

#img1 = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1/9/00020.jpg"
#img2 = "/Users/nathaninkawhich/DukeML/datasets/jester/20bn-jester-v1/9/00022.jpg"

#frame1 = cv2.imread(img1)
#frame2 = cv2.imread(img2)

#f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

f1_gray = np.zeros(shape=(50,50))
f1_gray[40,5] = 1
f1_gray[40,6] = 1
f1_gray[40,7] = 1
f1_gray[40,8] = 1
f1_gray[41,5] = 1
f1_gray[41,6] = 1
f1_gray[41,7] = 1
f1_gray[41,8] = 1

f2_gray = np.zeros(shape=(50,50))
f2_gray[15,10] = 1
f2_gray[15,11] = 1
f2_gray[15,12] = 1
f2_gray[15,13] = 1
f2_gray[16,10] = 1
f2_gray[16,11] = 1
f2_gray[16,12] = 1
f2_gray[16,13] = 1

flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 1, 20, 3, 5, 1.2, 0)

h_oflow = flow[...,0]
v_oflow = flow[...,1]


h_oflow = cv2.normalize(h_oflow, None, 0, 5, cv2.NORM_MINMAX)
h_oflow = cv2.normalize(h_oflow, None, 0, 5, cv2.NORM_MINMAX)

print h_oflow[40,6]
print v_oflow[40,6]
print h_oflow[40,7]
print v_oflow[40,7]
print ""
print h_oflow[45,10]
print v_oflow[45,10]

plt.subplot(221)
plt.imshow(f1_gray)
plt.subplot(222)
plt.imshow(f2_gray)
plt.subplot(223)
plt.imshow(h_oflow)
plt.subplot(224)
plt.imshow(v_oflow)

plt.show()

