import numpy as np 
import os
import skimage.io
import skimage.transform

#########################################################
# print_img
def print_img(img):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			print img[i,j]," ",
		print

#########################################################
# compute_stdev_map
# This function computes the standard deviation map called for
#	in the paper. It computes the standard deviation of each
#	n x n neighborhood in the input img.
#	Note: the img should have a border on it so the stdev map
#		has the same dimensionality as the original image
def compute_stdev_map(img, n):

	stdev_map = np.zeros(shape=[ (img.shape[0] - (n-1)), (img.shape[1]-(n-1)) ]) 
	for i in range(img.shape[0] - (n-1)):
		for j in range(img.shape[1] - (n-1)):
			#print "img[{},{}] = \n{}".format(i,j,img[i:i+n, j:j+n])
			#print "avg = {}".format(np.average(img[i:i+n, j:j+n]))
			#print "std = {}".format(np.std(img[i:i+n, j:j+n]))
			#print 

			stdev_map[i,j] = np.std(img[i:i+n, j:j+n])

	return stdev_map 

#########################################################
# compute_sensitivity_map
# Compute the sensitivity map that is called for in the paper.
#	This is calculated as the elementwise inverse of img.
def compute_sensitivity_map(img):
	sensitivity_map = np.reciprocal(img)
	return sensitivity_map



#########################################################
# apply_border
# This function returns a new image which is the input image
#	img with a n-pixel border of zeros around it. 
def apply_border(img, n):
	
	# Initialize a new matrix of zeros the size of the img plus the border
	new_img = np.zeros(shape=[ (img.shape[0]+2*n) , (img.shape[1]+2*n) ])

	# add the old image onto the top of the new zeros matrix
	new_img[n:n+img.shape[0], n:n+img.shape[1]] += img

	return new_img

#########################################################
# distance_fxn
# This is the D(X',X) function called for in the paper that quantifies
# 	the distance between the original image and the perturbed image
def distance_fxn(sensitivity_map, delta_map):
	assert(sensitivity_map.shape == delta_map.shape)
	return sensitivity_map.ravel().dot(delta_map.ravel())
	

#########################################################
# MAIN

# Inputs
input_img = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnistasjpg/testSample/img_1.jpg")
orig_class = 2
target_class = 3


# Read image as np array
# Image is already read in as 2-D so no need to remove the channel dimension
#img = skimage.io.imread(input_img)
#print "orig img shape: ",img.shape

# Randomly initialize a matrix representing an image (for testing only)
img = np.random.randint(0, 20, size=[5,5])
print "********** original img **********\n",img

# Apply the border to the image
border_size = 1
border_img = apply_border(img, border_size)
print "********** border img **********\n",border_img

print "img shape: ", border_img.shape

# Compute the stdev map with the bordered image and considering 
#	3x3 neighborhoods. Notice it is not an accident that we applied
#	a 1-px border and a 3x3 neighborhood as this combination will 
#	MAINTAIN the dimensionality of the original image. 
stdev_map = compute_stdev_map(border_img, 3)
print "********** stdev map **********\n",stdev_map

# Compute the sensitivity map from the paper which is the element wise inverse of stdev map
sensitivity_map = compute_sensitivity_map(stdev_map)
print "********** sensitivity map **********\n",sensitivity_map

# Initialize the delta map which is the same size as the original image. This
#	map holds the pixelwise perturbations of the original image
delta_map = np.zeros(shape=img.shape)

print "D(X',X) = ",distance_fxn(sensitivity_map, delta_map)










