##################################################################################
# NAI
#
# This script will provide a function that shows the full perturbation path
#	of an image from an optical flow. This script is self contained and will 
#	illustrate the process of:
#	1. Creating optical flow from 2 sequential frames of video
#	2. Perturb optical flow in some manner at some locations
#	3. Create 2 adversarial frames by applying the perturbations from the optical 
#		flow onto the original frames
#	4. Recalculate optical flow between the two adversarial frames to ensure
#		that the perturbations applied to the frames actually show up in the 
#		new optical flow
#
# 
##################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt 
import cv2
import os

# ***************************************************************
# Function to calculate dense optical flow between two adjacent frames
def calc_optical_flow(frame1, frame2):
  
	# Read in the images
	#frame1 = cv2.imread(img1)
	#frame2 = cv2.imread(img2)

	# Convert the images to grayscale
	f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	# Calculate dense optical flow
	# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
	# NATES CV2
	flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	# MATTS CV2
	#flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 15, 3, 5, 1.2, 0)

	# Separate horizontal and vertical pieces
	h_oflow = flow[...,0]
	v_oflow = flow[...,1]

	# Recenter to 127
	h_oflow += 127
	v_oflow += 127

	# Clip at the bounds [0,255]
	h_oflow[h_oflow < 0] = 0
	h_oflow[h_oflow > 255] = 255
	v_oflow[v_oflow < 0] = 0
	v_oflow[v_oflow > 255] = 255

	# Cast to integers
	h_oflow = np.rint(h_oflow)
	v_oflow = np.rint(v_oflow)

	# Return the oflow
	return h_oflow,v_oflow

##################################################################################
# Function: perturbed_oflow_to_images
#
# Inputs:
# - img1 - original frame 1
# - img2 - original frame 2
# - pert_oflow_h - the perturbed horizontal optical flow component
# - pert_oflow_v - the perturbed vertical optical flow component
# - perturb_locs - locations of the perturbations in the optical flow
#
# Outputs:
# - adv_img1 - perturbed image (first in sequence)
# - adv_img2 - perturbed image (second in sequence)
#
def perturbed_oflow_to_images(img1, img2, pert_oflow_h, pert_oflow_v, perturb_locs):

	# Size of cluster that will be placed on image
	# For now, this is a square with sides of length K
	K = 2

	# Make copy of img1 and img2, which will be perturbed and returned
	adv1 = np.copy(img1)
	adv2 = np.copy(img2)

	# For each loc in perturb_locs
	for loc in perturb_locs:

		# Location to perturb in the first image
		# x,y = loc
		row = loc[0]
		col = loc[1]

		# Location to perturb in the second image
		row2 = row + pert_oflow_v[row,col]
		col2 = col + pert_oflow_h[row,col]

		print "loc: ",loc
		print "row: ",row
		print "col: ",col
		print "row2: ",row2
		print "col2: ",col2

		# TODO: Bounds Checking!!!!!!
		# Make sure perturbing at row,col in img1 will not cause us to go out of bounds
		# Make sure perturbing the second image at [(row,col) + oflow(row,col)] will not be out of bounds
		
		# Place cluster of K pixels with color C onto img1 at loc
		for i in range(K):
			for j in range(K):
				adv1[int(row+i),int(col+j)] = img1[row,col]

		# Place cluster of K pixels with color C onto img2 at loc( row + oflow_v(row,col), col + oflow_h(row,col) )
		for i in range(K):
			for j in range(K):
				adv2[int(row2+i),int(col2+j)] = img1[row,col]

	# Return adversarial images
	return adv1,adv2



##################################################################################
# MAIN

# Inputs
i1 = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1/9/00012.jpg")
i2 = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1/9/00014.jpg")

# Read the images into numpy arrays
img1 = cv2.imread(i1)
img2 = cv2.imread(i2)

# Calculate Optical Flow
h_oflow,v_oflow = calc_optical_flow(img1, img2)

h_oflow -= 127
v_oflow -= 127

# Make copies of the optical flow to play with
pof_h = np.copy(h_oflow)
pof_v = np.copy(v_oflow)

# Find the magnitudes of movement given the h and v oflows mag(x,y) = sqrt( (h_oflow^2) + (v_oflow^2) )
magnitudes = np.sqrt((h_oflow)**2 + (v_oflow)**2)

# Find the top N locations of magnitude
N = 3
indices = np.argpartition(magnitudes.flatten(), -N)[-N:]
locs = np.vstack(np.unravel_index(indices, magnitudes.shape)).T

print "Perturbing at: ",locs

# Apply the N perturbations to optical flow field
for loc in locs:
	row = loc[0]
	col = loc[1]
	pof_h[row,col] *= -1
	pof_v[row,col] *= -1

# Reverse the optical flow perturbations onto two adversarial spatial images
pimg1,pimg2 = perturbed_oflow_to_images(img1,img2,pof_h,pof_v,locs)

# Recalculate optical flow on adversarial spatial images
p_h_oflow,p_v_oflow = calc_optical_flow(pimg1, pimg2)

##################################################################################
# Plotting results

# Plot Results 2x5 grid
# [1] - Plot original 1st frame
plt.subplot(3,4,1)
plt.imshow(img1)

# [2] - Plot original 2nd frame
plt.subplot(3,4,2)
plt.imshow(img2)

# [3] - Plot original horizontal optical flow
plt.subplot(3,4,3)
plt.imshow(h_oflow)

# [4] - Plot original vertical optical flow
plt.subplot(3,4,4)
plt.imshow(v_oflow)

# [5] - Plot Vector Field of original optical flow (!!!!!!!!)
plt.subplot(3,4,5)
plt.imshow(magnitudes)

# [6] - Plot Manually perturbed horizontal optical flow
plt.subplot(3,4,6)
plt.imshow(pof_h)

# [7] - Plot Manually perturbed vertical optical flow
plt.subplot(3,4,7)
plt.imshow(pof_v)

# [8] - Plot Vector Field of manually perturbed oflow (!!!!!!!!)
plt.subplot(3,4,8)
plt.imshow(magnitudes)

# [9] - Plot Adversarial 1st frame
plt.subplot(3,4,9)
plt.imshow(pimg1)

# [10] - Plot Adversarial 2nd frame
plt.subplot(3,4,10)
plt.imshow(pimg2)

# [11] - Plot h oflow between adversarial frames
plt.subplot(3,4,11)
plt.imshow(p_h_oflow)

# [12] - Plot v oflow between adversarial frames
plt.subplot(3,4,12)
plt.imshow(p_v_oflow)

# [13] - Plot vector flow of adversarial oflow (!!!!!!!!)
#plt.subplot(3,4,13)
#plt.imshow(magnitudes)

plt.show()




'''
##################################################################################
# Call perturbed_oflow_to_images

# Simulate input images
i1 = np.random.rand(8,8)
i2 = np.random.rand(8,8)

# Locations of perturbations
perturb_locs = [ [5,1] , [4,5] ]

# Create perturbed optical flow fields
pof_h = np.zeros(shape=i1.shape)
pof_v = np.zeros(shape=i1.shape)
pof_h[5,1] = -1
pof_v[5,1] = 1
pof_h[4,5] = -2
pof_v[4,5] = -3

p1,p2 = perturbed_oflow_to_images(i1,i2,pof_h,pof_v,perturb_locs)

plt.subplot(121)
plt.imshow(p1)
plt.subplot(122)
plt.imshow(p2)
plt.show()
'''