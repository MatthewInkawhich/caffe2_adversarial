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
import skimage.io
import skimage.transform
import operator

# location of input pb's
init_net_loc = "mnist_init_net.pb"
pred_net_loc = "mnist_predict_net.pb"

if len(sys.argv) != 2:
    print "Usage: mnist_test_jpg.py </path/to/img.jpg>"
    exit()
else:
	img_loc = sys.argv[1]

# Make sure the specified inputs exist
if ((not os.path.exists(img_loc)) or (not os.path.exists(init_net_loc)) or (not os.path.exists(pred_net_loc))):
	print "ERROR: An input was not found"
	exit()

# Bring in input image as float
orig_img = skimage.img_as_float(skimage.io.imread(img_loc)).astype(np.float32)
print "image shape: ",orig_img.shape

# Add an axis in the channel dimension
img = orig_img[np.newaxis, :, :].astype(np.float32)
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





