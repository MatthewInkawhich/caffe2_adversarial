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

# Warping images with optical flow
#https://stackoverflow.com/questions/44535091/convection-of-an-image-using-optical-flow
#https://www.mathworks.com/matlabcentral/answers/23708-using-optical-flow-to-warp-an-image
#http://pytorch.org/docs/master/_modules/torch/nn/functional.html#grid_sample
#https://stackoverflow.com/questions/17459584/opencv-warping-image-based-on-calcopticalflowfarneback

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

	print frame1.shape
	print frame1.max()
	print frame1.min()
	exit()

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

	# From beyond short snippits
	h_oflow[h_oflow < -40] = -40
	h_oflow[h_oflow > 40] = 40
	v_oflow[v_oflow < -40] = -40
	v_oflow[v_oflow > 40] = 40

	h_oflow = cv2.normalize(h_oflow, None, 0, 255, cv2.NORM_MINMAX)
	v_oflow = cv2.normalize(v_oflow, None, 0, 255, cv2.NORM_MINMAX)

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

	#print img1.shape
	#print img1.max()
	#print img1.min()
	#exit()

	# Size of cluster that will be placed on image
	# For now, this is a square with sides of length K
	K = 10

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

def print_vector_field(h_oflow, v_oflow, title,downsample=1):
    # X: x-coordinates of the arrow tails
    # Y: y-coordinates of the arrow tails
    # U: x components of arrow vectors
    # V: y components of arrow vectors
    image_height, image_width = h_oflow.shape

    # Create meshgrid for coordinate resemblance
    Y, X = np.mgrid[0:image_height, 0:image_width]

    # Initialize U and V as the same size as image
    U = np.zeros((image_height,image_width))
    V = np.zeros((image_height,image_width))

    # Fill U and V with x and y vector components from h_img and v_img
    for i in range(image_height):
            for j in range (image_width):
                if i%downsample==0 and j%downsample==0:
                    U[i,j] = h_oflow[i,j]
                    V[i,j] = v_oflow[i,j]

    # Plot optical flow flow field
    plt.figure()
    plt.title(title)
    plt.quiver(X, Y, U, V, scale=1, units='xy')
    plt.xlim(-1, image_width)
    plt.ylim(-1, image_height)
    plt.gca().invert_yaxis()
    plt.show(block=False)


##################################################################################
# MAIN

# Inputs
i1 = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1/9/00014.jpg")
i2 = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1/9/00015.jpg")

# Read the images into numpy arrays
img1 = cv2.imread(i1)
img2 = cv2.imread(i2)

# Calculate Optical Flow
h_oflow,v_oflow = calc_optical_flow(img1, img2)


# Make copies of the optical flow to play with
pof_h = np.copy(h_oflow)
pof_v = np.copy(v_oflow)

# Find the magnitudes of movement given the h and v oflows mag(x,y) = sqrt( (h_oflow^2) + (v_oflow^2) )
magnitudes = np.sqrt((h_oflow)**2 + (v_oflow)**2)

# Find the top N locations of magnitude
N = 20
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

f, axarr = plt.subplots(2,5,figsize=(10,10))

# Plot Results 2x5 grid
# [1] - Plot original 1st frame
axarr[0,0].set_title("Orig Frame 1")
axarr[0,0].imshow(img1)
axarr[0,0].axis("off")
#axarr[0,0].tight_layout()

# [2] - Plot original 2nd frame
axarr[0,1].set_title("Orig Frame 2")
axarr[0,1].imshow(img2)
axarr[0,1].axis("off")
#axarr[0,1].tight_layout()

# [3] - Plot original horizontal optical flow
axarr[0,2].set_title("Orig H Oflow")
axarr[0,2].imshow(h_oflow,cmap='gray')
axarr[0,2].axis("off")
#axarr[0,2].tight_layout()

# [4] - Plot original vertical optical flow
axarr[0,3].set_title("Orig V Oflow")
axarr[0,3].imshow(v_oflow,cmap='gray')
axarr[0,3].axis("off")
#axarr[0,3].tight_layout()

# [5] - Plot Manually perturbed horizontal optical flow
axarr[0,4].set_title("Manually Pert H Oflow")
axarr[0,4].imshow(pof_h)
axarr[0,4].axis("off")
#axarr[0,4].tight_layout()

# [6] - Plot Manually perturbed vertical optical flow
axarr[1,0].set_title("Manually Pert V Oflow")
axarr[1,0].imshow(pof_v)
axarr[1,0].axis("off")
#axarr[1,0].tight_layout()

# [7] - Plot Adversarial 1st frame
axarr[1,1].set_title("Adversarial Frame 1")
axarr[1,1].imshow(pimg1)
axarr[1,1].axis("off")
#axarr[1,1].tight_layout()

# [8] - Plot Adversarial 2nd frame
axarr[1,2].set_title("Adversarial Frame 2")
axarr[1,2].imshow(pimg2)
axarr[1,2].axis("off")
#axarr[1,2].tight_layout()

# [9] - Plot h oflow between adversarial frames
axarr[1,3].set_title("Calculated Pert H Oflow")
axarr[1,3].imshow(p_h_oflow)
axarr[1,3].axis("off")
#axarr[1,3].tight_layout()

# [10] - Plot v oflow between adversarial frames
axarr[1,4].set_title("Calculated Pert V Oflow")
axarr[1,4].imshow(p_v_oflow)
axarr[1,4].axis("off")
#axarr[1,4].tight_layout()

# Create oflow vector field
print_vector_field(h_oflow, v_oflow, "Original OFlow Vector Field", downsample=6)

# Create oflow vector field
print_vector_field(p_h_oflow, p_v_oflow, "Perturbed OFlow Vector Field", downsample=6)

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
