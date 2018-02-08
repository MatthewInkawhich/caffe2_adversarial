# MatthewInkawhich
# This script runs inference on a single stack of optical flow images

import sys
from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import cv2
import glob


########################################################################
# CONFIGS
########################################################################
# location of input pb's
init_net_loc = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_NO_LMDB_output', 'mstar_NO_LMDB_init_net.pb')
pred_net_loc = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_NO_LMDB_output', 'mstar_NO_LMDB_predict_net.pb')
dictionary_file = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'TrainDictionary_5class.txt')
seq_size = 10

########################################################################
# FUNCTIONS
########################################################################
def crop_center(img, new_height, new_width):
    orig_height, orig_width, _ = img.shape
    startx = (orig_width//2) - (new_width//2)
    starty = (orig_height//2) - (new_height//2)
    return img[starty:starty+new_height, startx:startx+new_width]

def resize_image(img, new_height, new_width):
    h, w, _ = img.shape
    if (h < new_height or w < new_width):
        img_data_r = imresize(img, (new_height, new_width))
    else:
        img_data_r = crop_center(img, new_height, new_width)
    return img_data_r

def handle_greyscale(img):
    img = img[:,:,0]
    #img = np.expand_dims(img, axis=2)
    return img

def create_oflow_stack(seq):
	# Given seq of ordered jpgs for the optical flow, read them into a numpy array
	# seq = [ \path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ... ]

	#print "in create oflow stack"
	oflow_stack = np.zeros(shape=(20,100,100))

	# For each of the images in the sequence (which are contiguous optical flow images)
	for i in range(len(seq)):

		# Read the image as a color image (BGR) into a numpy array as 32 bit floats
		of_img = cv2.imread(seq[i]).astype(np.float32)
		#print "Shape after reading in : {}".format(of_img.shape)

		# Resize the image to 3x100x100
		of_img = resize_image(of_img, 100, 100)
		#print "Shape after Resizing : {}".format(of_img.shape)

		of_img = handle_greyscale(of_img)
		#print "Shape after greyscale : {}".format(of_img.shape)

		oflow_stack[i,:,:] = of_img

		#print "Printing"
		#print oflow_stack
	return oflow_stack



########################################################################
# MAIN
########################################################################

# Make sure the specified inputs exist
if ((not os.path.exists(init_net_loc)) or (not os.path.exists(pred_net_loc))):
	print "ERROR: An input was not found"
	exit()

# Find label corresponding to desired vid_number
df = open(dictionary_file)
vid_number = 4060
for line in df:
    path = line.split()[0]
    label = line.split()[1]
    if int(path.split('/')[-1]) == vid_number:
        print "path:", path, "label:", label
        break

# Obtain all oflow jpgs in the given video directory
of_jpgs = glob.glob(os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', str(vid_number)) + "/*.jpg")

# Sort array by h & v, then by sequence number
of_jpgs.sort(key=lambda x: x.split('/')[-1].split('_')[-1].split('.')[0])
of_jpgs.sort(key=lambda x: int(x.split('/')[-1].split('_')[3]))

print of_jpgs
print "\n\n"
# Extract sequence of oflow jpgs to stack
stack_seq = of_jpgs[0:seq_size*2]

print stack_seq

stack = create_oflow_stack(stack_seq)
print stack.shape


# # Bring in input image as float
# orig_img = cv2.imread(img_loc).astype(np.float32)
# # resize image to desired dimensions
# orig_img = resize_image(orig_img, desired_h, desired_w)
# orig_img[:,:,(2,1,0)]
# orig_img = orig_img/255
# print(orig_img)
# img = orig_img
# # handle grayscale
# if ((img[:,:,0] == img[:,:,1]).all() and (img[:,:,0] == img[:,:,2]).all()):
#     img = handle_greyscale(img)
# # HWC -> CHW
# img = np.transpose(img, (2,0,1))
#
# print "image shape: ",img.shape
#
# # Add an axis in the channel dimension
# #img = orig_img[np.newaxis, :, :].astype(np.float32)
# img = img[np.newaxis, :, :, :].astype(np.float32)
# print "NCHW: ", img.shape
#
#
# # Bring up the network from the .pb files
# with open(init_net_loc) as f:
#     init_net = f.read()
# with open(pred_net_loc) as f:
#     predict_net = f.read()
#
# p = workspace.Predictor(init_net, predict_net)
#
# # run the net and return prediction
# results = p.run([img])
#
# print "results: ",results
#
# # turn it into something we can play with and examine which is in a multi-dimensional array
# results = np.asarray(results)
# print "results shape: ", results.shape
#
# results = results[0,0,:]
# print "results shape: ", results.shape
#
# max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))
#
# print "Prediction: ", max_index
# print "Confidence: ", max_value
#
# pyplot.figure()
# pyplot.imshow(orig_img, cmap='gray')
# pyplot.show()
