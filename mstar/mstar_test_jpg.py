# NAI
# This script takes the pretrained mnist pb files created with mnist_train_and_save.
#   It loads the nets and creates a predictor from the init and predict nets.
#   This script closely follows the loading pretrained models example from caffe2
# This script accepts one input which is a single mnist image to test

# ex. $ python mnist_test_jpg.py ~/DukeML/datasets/mnist/mnistasjpg/testSample/img_1.jpg

# import dependencies
print "Import Dependencies..."
import sys
from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
# import skimage.io
# import skimage.transform
import cv2
import operator


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
    img = np.expand_dims(img, axis=2)
    return img


desired_h = 64
desired_w = 64

# location of input pb's
init_net_loc = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_NO_LMDB_output', 'mstar_NO_LMDB_init_net.pb')
pred_net_loc = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_NO_LMDB_output', 'mstar_NO_LMDB_predict_net.pb')

if len(sys.argv) != 2:
    print "Usage: mstar_test_jpg.py </path/to/img.jpg>"
    exit()
else:
	img_loc = sys.argv[1]

# Make sure the specified inputs exist
if ((not os.path.exists(img_loc)) or (not os.path.exists(init_net_loc)) or (not os.path.exists(pred_net_loc))):
	print "ERROR: An input was not found"
	exit()



# Bring in input image as float
#orig_img = skimage.img_as_float(skimage.io.imread(img_loc)).astype(np.float32)
orig_img = cv2.imread(img_loc).astype(np.float32)
# resize image to desired dimensions
orig_img = resize_image(orig_img, desired_h, desired_w)
orig_img[:,:,(2,1,0)]
orig_img = orig_img/255
print(orig_img)
img = orig_img
# handle grayscale
if ((img[:,:,0] == img[:,:,1]).all() and (img[:,:,0] == img[:,:,2]).all()):
    img = handle_greyscale(img)
# HWC -> CHW
img = np.transpose(img, (2,0,1))

print "image shape: ",img.shape

# Add an axis in the channel dimension
#img = orig_img[np.newaxis, :, :].astype(np.float32)
img = img[np.newaxis, :, :, :].astype(np.float32)
print "NCHW: ", img.shape


# Bring up the network from the .pb files
with open(init_net_loc) as f:
    init_net = f.read()
with open(pred_net_loc) as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)

# run the net and return prediction
results = p.run([img])

print "results: ",results

# turn it into something we can play with and examine which is in a multi-dimensional array
results = np.asarray(results)
print "results shape: ", results.shape

results = results[0,0,:]
print "results shape: ", results.shape

max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))

print "Prediction: ", max_index
print "Confidence: ", max_value

pyplot.figure()
pyplot.imshow(orig_img, cmap='gray')
pyplot.show()
