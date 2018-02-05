# NAI
# This is the first version of the genetic adversarial attack based on the perceptual
# 	attack. The goal of this attack is to more inteligently/quickly generate 
#	adversarial examples with very low visual pertrubations.

# TODO/THINKABOUT:
# - Why would the confidence of the target class ever decrease? MUST FIX THIS
# - (OK) Must take into account the SIGN of the GAP. We will NOT perturb when the gap
#	 	increases as a result of the perturbation. This means that we may not always
#		perturb the same number of pixels in each generation
# - Add the probabilistic selecting
# - Treat the perturbations as a magnitude and every time we perturb we test in
#		both directions and pick the direction that works the best 
# - Add other procreate functions
# - Add distance in equation of fitness
# - Instead of randomly seeding, what if I find high variance zones to start with

import numpy as np 
import random
import os
import skimage.io
import skimage.transform
from caffe2.python import workspace
import operator
from operator import itemgetter
import matplotlib.pyplot as plt

PREDICTION_COUNT = 0

###################################################################################
# Generate Citizens
def generate_random_citizens(N, delta_max, x_max, y_max):
	
	# Randomly initialize N citizens and return a list of such citizens
	# 	Each member is a (x,y) coord and a delta
	# The bounds for the ints used for x and y correspond to 3x3 neighboorhood calculations
	Population = []

	for i in range(N):
		
		# Generate a UNIFORM random integer x in the bounds [1, x_max-1)
		x = np.random.randint(1,x_max-1)

		# Generate a UNIFORM random integer y in the bounds [1, y_max-1)
		y = np.random.randint(1,y_max-1)

		# Generate a UNIFORM random delta in the bounds [-delta_max, +delta_max]
		delta = random.uniform(0, delta_max)
		#delta = delta_max

		# Append this citizen to the population list
		Population.append([x,y,delta])

	return Population


###################################################################################
# Print Population
def print_population(P):
	for p in P:
		print "[ ({}, {}) , {} ]".format(p[0], p[1], p[2])

###################################################################################
# apply_border
# This function returns a new image which is the input image
#	img with a n-pixel border of zeros around it. 
def apply_border(img, n):
	
	# Initialize a new matrix of zeros the size of the img plus the border
	new_img = np.zeros(shape=[ (img.shape[0]+2*n) , (img.shape[1]+2*n) ])

	# add the old image onto the top of the new zeros matrix
	new_img[n:n+img.shape[0], n:n+img.shape[1]] += img

	return new_img

###################################################################################
# predict
# Input a predictor object and an image. Here the image is just 
#	a 2-D matrix so we must add the 3rd and 4th dimensions before
#	running the inference
def predict(predictor_obj, img):
	# Reshape the image for use with the classifier
	img = img[np.newaxis, :, :].astype(np.float32)
	img = img[np.newaxis, :, :, :].astype(np.float32)

	results = predictor_obj.run([img])

	# turn it into something we can play with and examine which is in a multi-dimensional array
	results = np.asarray(results)
	#print "results shape: ", results.shape

	results = results[0,0,:]

	#print "results: ",results

	# Increment the global counter
	global PREDICTION_COUNT 
	PREDICTION_COUNT += 1

	return results


###################################################################################
# gap_fxn
def gap_fxn(predictor_obj, img, target_class):

	# Run inference on the input img to get the vector of results
	results = predict(predictor_obj, img)

	# Extract the probability of the target class: P_t
	P_t = results[target_class]

	# Remove P_t and get the next highest probability
	r = np.delete(results,[target_class])

	# max probability among other classes
	max_P_i = np.amax(r)

	return -1*(P_t-max_P_i)

#########################################################
# compute_sensitivity_map
# Compute the sensitivity map that is called for in the paper.
#	This is calculated as the elementwise inverse of img.
def compute_sensitivity_map(img):
	sensitivity_map = np.reciprocal(img)
	return sensitivity_map

###################################################################################
# compute_stdev_map
# This function computes the standard deviation map called for
#	in the paper. It computes the standard deviation of each
#	3x3 neighborhood centered at the locations of the population.
def compute_stdev_map(img, n, P):

	stdev_map = np.zeros(shape=[ img.shape[0], img.shape[1] ]) 

	# For each citizen in the population, compute the stdev of its neighborhood
	for citizen in P:
		
		i = citizen[0]
		j = citizen[1]
		
		#print "Evaluating stdev: ({}, {})".format(i,j)

		# Compute the stddev of that neighborhood and add it to the stdev_map
		stdev_map[i,j] = np.std(img[i-1:i+2, j-1:j+2])				

	return stdev_map 

###################################################################################
# perturb_priority
# Calculates the perturb priority for every pixel in the img and
#	returns a matrix of the priorities of the same dimensionality
#	of the input img
def perturb_priority(predictor_obj, img, Population, target_class, do_not_touch ):

	# Calculate the sparse STD map for the img only at the locations specified in the population.
	# Note: currently the 3 means nothing here
	stdev_map = compute_stdev_map(img, 3, Population)
	#print "************* stdev map *************\n",stdev_map

	# Calculate the Sensitivity map given the std map. There may be divide by zeros
	#	here but that is ok because we want those areas to have 'inf' sensitivity
	sens_map = compute_sensitivity_map(stdev_map)
	#print "************* sens map *************\n",sens_map

	# Initialize perturb priority map (which will be returned)
	perturb_priority_map = np.zeros(shape=img.shape)

	# Run the gap fxn for the input img. This serves as a reference when computing gradient of gap_fxn
	orig_gap = gap_fxn(predictor_obj, img, target_class)

	# For each pixel in the population
	for citizen in Population:

		i = citizen[0]
		j = citizen[1]

		# Make a temporary copy of the input img to perturb
		tmp_img = np.copy(img)
		# Apply unit perturbation to temporary img at current pixel location
		# Apply the perturbation in the direction of the sign of the delta
		if np.sign(citizen[2]) == 0:
			tmp_img[i,j] += 1
		else:
			tmp_img[i,j] -= 1
		# Run the gap fxn for the newly perturbed tmp_img
		tmp_gap = gap_fxn(predictor_obj, tmp_img, target_class)
		# Read the sensitivity value of the current pixel from the precomputed sensitivity map
		tmp_sens = sens_map[i,j]
		# Calculate the gradient of the gap fxn as the difference between the tmp gap and the orig gap
		gradient_gap = tmp_gap - orig_gap
		# Insert perturb priority into map at current pixel
		perturb_priority_map[i,j] = gradient_gap/tmp_sens
		
		#print "pert priority [{}, {}] = {}".format(i,j,perturb_priority_map[i,j])

	# for each of the do not touch locations, set there perturb priority very negative so they
	# will not get picked to be perturbed
	for loc in do_not_touch:
		perturb_priority_map[loc[0], loc[1]] = -99999

	return perturb_priority_map

###################################################################################
# get top performers
def get_top_performers(num, pop):
	
	top_performers = []
	for p in pop:
		# If the fitness is positive, look to potentially add it
		if p[3] > 0:
			exists = False
			# Check if current location already exists in top_performers
			for t in top_performers:
				if ((p[0] == t[0]) and (p[1] == t[1])):
					exists = True
					break
			# If the location exists, check the next one
			if exists == True:
				continue;

			# If it does not exist, add it to top performers
			top_performers.append(p)

			# Check if we are done adding top performers
			if len(top_performers) == num:
				break
		# Since pop is sorted here, as soon as we hit a negative value we exit because
		#	perturbing at a location with negative fitness would have negative effects
		else:
			break

	return top_performers

###################################################################################
# procreate
def procreate(P1, P2):
	
	# Option1: Midpoint rule
	new_x = int((P1[0]+P2[0])/2)
	new_y = int((P1[1]+P2[1])/2)
	# Take the delta from the one that performed better
	new_delta = 0.
	if P1[3] > P2[3]:
		new_delta = P1[2]
	else:
		new_delta = P2[2]

	#print "Procreate: P1.[{},{},{}] + P2.[{},{},{}] = C.[{},{},{}]".format(pop[i1][0],pop[i1][1],pop[i1][2],pop[i2][0],pop[i2][1],pop[i2][2], new_x, new_y, new_delta)

	return [new_x, new_y, new_delta]

###################################################################################
# MAIN ALGORITHM

# Inputs
init_net_loc = "../mnist_init_net.pb"
pred_net_loc = "../mnist_predict_net.pb"
input_img = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnistasjpg/testSample/img_239.jpg")
orig_class = 8 # correct class of the input image
target_class = 3 # target class of the adversarial image
Population_size = 100 # Size of the population
MAX_GENERATIONS = 1000 # Maximum number of generations
PERTURBATIONS_PER_ITER = 5 # The number of perturbations to perform at each iteration
delta_max = .02 # Maximum magnitude of perturbation that can be applied to a pixel in a single iteration
num_randoms = 10 # The number of randomly generated citizens in every generation

# Read in the image
img = skimage.img_as_float(skimage.io.imread(input_img)).astype(np.float32)
#img = np.random.randint(20, size=(5,5))

#print img 

x_max = img.shape[0]
y_max = img.shape[1]

# save a copy of the original image for later
orig_img = np.copy(img)

# Initialize the predictor
with open(init_net_loc) as f:
    init_net = f.read()
with open(pred_net_loc) as f:
    predict_net = f.read()

# Initialize the predictor which will repetedly be used to run inference
predictor = workspace.Predictor(init_net, predict_net)

# Randomly generate the initial population
Population = generate_random_citizens(Population_size, delta_max, x_max, y_max)
#print_population(Population)

# Initialize the list where we will store pixel location that are at the bounds
do_not_touch = []

# While we have not reached the max number of iterations
cnt = 0
while cnt < MAX_GENERATIONS:

	# Compute the perturbation priority of each pixel in the population and all
	#	of the pixels within an n x n neighborhood of it. If the current location is 
	# 	in the do not touch list, automatically set the P.P of that location to zero.
	# 	This is a sparse matrix as we do not need to compute the P.P. for a pixel
	#	that will never be considered for applying a perturbation.
	perturbation_priority_map = perturb_priority(predictor, img, Population, target_class, do_not_touch)

	# *** Calculate the fitness of each pixel in population
	# Option 1: Fitness = Perturbation Priority
	population_with_fitness = []
	for p in Population:
		population_with_fitness.append([p[0],p[1],p[2],perturbation_priority_map[p[0],p[1]]] )
	#print "********** Population with fitness ***********\n",population_with_fitness

	# Sort the population based on fitnesses 
	population_with_fitness.sort(key=lambda x : x[3], reverse=True)
	#print "********** Sorted Population with fitness ***********\n",population_with_fitness

	# Starting from the best fitness, add the first N UNIQUE locations to top performers list
	#	This will make sure that we are not unintentionally perturbing the same location twice
	# 	which could potentially have a cancellation effect.
	top_performers = get_top_performers(PERTURBATIONS_PER_ITER, population_with_fitness)
	#print "********** Top Performers ***********\n",top_performers

	# For the top performers, perturb the image at the specified location by delta. Check
	# 	if any of the perturbations are at the bounds of the image. If so, perturb to the 
	#	bound and add that location to the do not touch list.
	for tp in top_performers:

		# Check if the perturbation will push the value outside of the bound
		if ((img[tp[0],tp[1]] + tp[2]) > 1):
			# add the perturbation that will put it to 1 exactly
			img[tp[0],tp[1]] = 1
			# add this location to the do_not_touch_locs list so we know not to touch this location again
			do_not_touch.append([tp[0],tp[1]])
			print "Added {} to do not touch!".format([tp[0],tp[1]])

		# Check if the perturbation will push the value outside of the bound
		elif ((img[tp[0],tp[1]] + tp[2]) < 0):
			# add the perturbation that will put it to 1 exactly
			img[tp[0],tp[1]] = 0
			# add this location to the do_not_touch_locs list so we know not to touch this location again
			do_not_touch.append([tp[0],tp[1]])
			print "Added {} to do not touch!".format([tp[0],tp[1]])

		# If it will not cause an overshoot, add the perturbation as normal	
		else:
			img[tp[0],tp[1]] += tp[2] 

	# Check if the new image is a success
	tmp_pred = predict(predictor,img)
	pred = np.argmax(tmp_pred)

	print "Iter: {}, Pred: {}, Conf: {}, Conf_t: {}".format(cnt, pred, tmp_pred[pred], tmp_pred[target_class])

	if(pred == target_class):
		print "*******************************\nSUCCESS"
		print "Num Predictions: ",PREDICTION_COUNT
		print "Max value in adv image: ",img.max()
		print "Min value in adv image: ",img.min()

		plt.figure()
		plt.subplot(1,2,1)
		plt.title("Original Image")
		plt.axis('off')
		plt.imshow(orig_img, cmap='gray')
		plt.subplot(1,2,2)
		plt.title("Adversarial Image")
		plt.axis('off')
		plt.imshow(img, cmap='gray')
		plt.show()

		exit()

	# Initialize the new generation
	new_population = []

	# Automatically advance the top K performers of this generation to the new generation
	# 	This is not redundant because there will be new sensitivity and P.P.
	#	values because we just perturbed
	# (?) - Should I generate a new random delta in the same direction for each of these
	for tp in top_performers:
		new_population.append([tp[0],tp[1],tp[2]])

	# Create children by selecting parents to mate
	for i in range(Population_size - len(new_population) - num_randoms):

		# Probabilistically select a pair of citizens to reproduce
		# TODO: fix this. I am just drawing randomly from top 60% right now
		# Generate a random number from 0 to int(.6*Population_size)
		r1 = np.random.randint(int(.6*Population_size))
		r2 = np.random.randint(int(.6*Population_size))
		# Make sure r2 != r1
		while(r2 == r1):
			r2 = np.random.randint(int(.6*Population_size))

		# Generate the child from the two parents
		child = procreate(population_with_fitness[r1], population_with_fitness[r2])

		# Add the child to the next generation
		new_population.append(child)

	# For the rest, generate some randoms
	rands = generate_random_citizens(Population_size-len(new_population), delta_max, x_max, y_max)
	for r in rands:
		new_population.append(r)

	# Make sure the new generation is the same size as the old generation
	assert(len(new_population) == Population_size)

	# Kill off the old population
	Population = new_population

	# Increment the iteration count
	cnt += 1

	#exit()




