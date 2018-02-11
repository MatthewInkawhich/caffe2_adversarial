##################################################################################
# NAI
#
# This script will provide a function that places perturbations from an 
#	adversarial optical flow onto two original spatial frames such that
#	if the optical flow was recomputed on the two perturbed spatial frames, 
#	the perturbations would appear in the recalculated optical flow.
#
# Note: This function works on a single optical flow image (h & v). To use with
#	a stack, you must first split up the stack
##################################################################################

import numpy as np
import random
import matplotlib.pyplot as plt 

##################################################################################
# Function: images_from_perturbed_optical_flow
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
def images_from_perturbed_optical_flow(img1, img2, pert_oflow_h, pert_oflow_v, perturb_locs):

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
		x = loc[0]
		y = loc[1]

		# Location to perturb in the second image
		x2 = x + pert_oflow_v[x,y]
		y2 = y + pert_oflow_h[x,y]

		print "loc: ",loc
		print "x: ",x
		print "y: ",y
		print "x2: ",x2
		print "y2: ",y2

		# TODO: Bounds Checking!!!!!!

		# Make sure perturbing at x,y in img1 will not cause us to go out of bounds
		#assert( (x > (K // 2)) and (x < (img1.size[0] - (K // 2))) ) 
		#assert( (y > (K // 2)) and (y < (img1.size[1] - (K // 2))) ) 

		# Make sure perturbing the second image at [(x,y) + oflow(x,y)] will not be out of bounds
		
		# Place cluster of K pixels with color C onto img1 at loc
		for i in range(K):
			for j in range(K):
				adv1[int(x+i),int(y+j)] = img1[x,y]

		# Place cluster of K pixels with color C onto img2 at loc( x + oflow_v(x,y), y + oflow_h(x,y) )
		for i in range(K):
			for j in range(K):
				adv2[int(x2+i),int(y2+j)] = img1[x,y]

	# Return adversarial images
	return adv1,adv2

##################################################################################
# Call the function

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

p1,p2 = images_from_perturbed_optical_flow(i1,i2,pof_h,pof_v,perturb_locs)

plt.subplot(121)
plt.imshow(p1)
plt.subplot(122)
plt.imshow(p2)
plt.show()
