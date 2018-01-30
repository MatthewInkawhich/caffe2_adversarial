import numpy as np 
import os
import skimage.io
import skimage.transform
from caffe2.python import workspace
import operator
from operator import itemgetter
import matplotlib.pyplot as plt

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
# predict
# Input a predictor object and an image. Here the image is just 
#	a 2-D matrix so we must add the 3rd and 4th dimensions before
#	running the inference
def predict(predictor_obj, img):
	# Reshape the image for use with the classifier
	img = img[np.newaxis, :, :].astype(np.float32)
	img = img[np.newaxis, :, :, :].astype(np.float32)

	results = p.run([img])

	# turn it into something we can play with and examine which is in a multi-dimensional array
	results = np.asarray(results)
	#print "results shape: ", results.shape

	results = results[0,0,:]
	#print "results shape: ", results.shape

	#print "results: ",results

	return results

#########################################################
# gap_fxn
def gap_fxn(predictor_obj, img, target_class):

	# Run inference on the input img to get the vector of results
	results = predict(predictor_obj, img)

	#print results

	# Extract the probability of the target class: P_t
	P_t = results[target_class]

	# Remove P_t and get the next highest probability
	r = np.delete(results,[target_class])
	r = np.sort(r)

	# max probability among other classes
	max_P_i = r[-1]

	#print "P_t: ", P_t 
	#print "max(P_i): ", max_P_i
	#print "Gap = ", P_t-max_P_i

	return P_t-max_P_i

#########################################################
# perturb_priority
# Calculates the perturb priority for every pixel in the img and
#	returns a matrix of the priorities of the same dimensionality
#	of the input img
def perturb_priority(predictor_obj, img, target_class):

	# Run the gap fxn for the input img
	orig_gap = gap_fxn(predictor_obj, img, target_class)

	# Calculate the SD map for the img
	# We add a 1-px border of zeros around the input image because the 
	#	compute_stdev_map function computes the stdev within 3x3 neighborhoods
	#	so the dimensionality is reduced. With this combination, the stdev_map
	#	is the same dimensionality as the input img.
	border_img = apply_border(img, 1)
	stdev_map = compute_stdev_map(border_img, 3)

	# Calculate the Sensitivity map given the std map
	sens_map = compute_sensitivity_map(stdev_map)

	# Initialize perturb priority map (which will be returned)
	perturb_priority_map = np.zeros(shape=img.shape)

	# For each pixel in the img
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			# Make a temporary copy of the input img
			tmp_img = np.copy(img)
			# Apply unit perturbation to temporary img at current pixel location
			tmp_img[i,j] += 1
			# Run the gap fxn for the newly perturbed tmp_img
			tmp_gap = gap_fxn(predictor_obj, tmp_img, target_class)
			# Read the sensitivity value of the current pixel from the precomputed sensitivity map
			tmp_sens = sens_map[i,j]
			# Calculate the gradient of the gap fxn as the difference between the tmp gap and the orig gap
			gradient_gap = tmp_gap - orig_gap
			# Insert perturb priority into map at current pixel
			perturb_priority_map[i,j] = gradient_gap/tmp_sens
			#print "pert priority [{}, {}] = {}".format(i,j,perturb_priority_map[i,j])

	return perturb_priority_map, sens_map

def get_coords_of_n_largest(mat, n):
	# Assert mat is a 2d matrix
	assert(len(mat.shape) == 2)
	flat = []
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			flat.append([i,j,mat[i,j]])
	#print flat
	# Note: each entry in flat is [x_coord, y_coord, value]
	# sort the flat list according to the value at that coordinate
	flat.sort(key=lambda x : x[2])
	# Extract the n_largest entries from the flat list
	n_largest = flat[-n:]
	# Just take the coordinates of the n_largest, dont care about the values here
	coords = [x[0:2] for x in n_largest]
	return coords 

#########################################################
# MAIN
#########################################################

# Inputs
init_net_loc = "../mnist_init_net.pb"
pred_net_loc = "../mnist_predict_net.pb"
input_img = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnistasjpg/testSample/img_6.jpg")
orig_class = 7 # correct class of the input image
target_class = 9 # target class of the adversarial image
D_max = 20 # max allowed human perceptual distance
m = 20 # number of pixels perturbed in each iteration
delta = 0.01 # perturbation magnitude
MAX_ITERS = 300

#########################################################
# Read in input image

# Read image as np array
# Image is already read in as 2-D so no need to remove the channel dimension
img = skimage.img_as_float(skimage.io.imread(input_img)).astype(np.float32)
#print "orig img shape: ",img.shape

#########################################################
# Initiailize the predictor

# Bring up the network from the .pb files
with open(init_net_loc) as f:
    init_net = f.read()
with open(pred_net_loc) as f:
    predict_net = f.read()

# Initialize the predictor which will repetedly be used to run inference
p = workspace.Predictor(init_net, predict_net)

# Matrix to save all of the perturbations
# At the end, to construct the adv image, this will be added to the original image
delta_map = np.zeros(shape=img.shape)

# save the original image
orig_img = np.copy(img)

cnt = 0
pred = orig_class

while(cnt < MAX_ITERS):

	# Calculate the perturbation priority of all pixels for the current image
	perturb_priority_map, sensitivity_map = perturb_priority(p, img, target_class)

	# Get the coordinates of the largest n perturbation priorities
	selected_pixels = get_coords_of_n_largest(perturb_priority_map, m)

	# Perturb the selected pixel coordinates
	for s in selected_pixels:
		delta_map[s[0],s[1]] += delta
		img[s[0],s[1]] += delta 

	# Calculate the new distance
	dist = distance_fxn(sensitivity_map, delta_map)

	# Check for success
	tmp_pred = predict(p,img)
	pred = np.argmax(tmp_pred)

	print "Iter: {}, Prediction: {}, Confidence: {}".format(cnt, pred, tmp_pred[pred])

	if(pred == target_class):
		print "SUCCESS"

		print img.max()
		print img.min()
		

		plt.figure()
		plt.subplot(1,2,1)
		plt.imshow(orig_img)
		plt.subplot(1,2,2)
		plt.imshow(img)
		plt.show()
		exit()

	cnt += 1

exit()










