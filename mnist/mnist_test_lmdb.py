# NAI
# This script uses the pb files created with the mnist_train_and_save script.
#   It loads the nets from the pb files into a model helper, it then adds the data
#   op and the accuracy op to the model helper and runs training on the mnist test lmdb.
#   The point of this is to show how to load a pretrained model and test it with an lmdb.

# import dependencies
print "Import Dependencies..."
from matplotlib import pyplot
import numpy as np 
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter

##################################################################################
# Get inputs

# location of inputs
TEST_LMDB = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnist-lmdb/mnist-test-nchw-lmdb")
INIT_NET = "mnist_init_net.pb"
PREDICT_NET = "mnist_predict_net.pb"

# Make sure the specified inputs exist
if ((not os.path.exists(TEST_LMDB)) or (not os.path.exists(INIT_NET)) or (not os.path.exists(PREDICT_NET))):
	print "ERROR: An input was not found"
	exit()

##################################################################################
# Create a model object (using model helper)

# We will populate this model object with the nets from the .pb's then run testing on it
arg_scope = {"order": "NCHW"}
test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)

##################################################################################
# Add the input ('data') stuff to the model (just as in training)

# The pretrained model we are importing expects the input to the network to be a blob named 'data'
# Equivalent to the AddInput() fxn from the MNIST tutorial
# Go read the data from the lmdb and format it properly
# Use TensorProtosDBInput to read the lmdb and create a 'data_uint8' blob and a 'label' blob
data_uint8, label = test_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=100, db=TEST_LMDB, db_type='lmdb')
# cast the 8-bit data to floats
data = test_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
# scale data from [0,255] -> [0,1]
data = test_model.Scale(data,data,scale=float(1./256))
# (?) enforce a stopgradient because we do not need the gradient of the data for the backward pass
#data = test_model.StopGradient(data,data)
# Here, we have access to the data and the labels
# the data is of shape [batch_size,num_channels,width,height]
# for mnist the shape is [bsize, 1, 28, 28]

##################################################################################
# Add the pretrained model stuff (info from the pbs) to the model helper object

# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
test_model.param_init_net = test_model.param_init_net.AppendNet(tmp_param_net)

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
test_model.net = test_model.net.AppendNet(tmp_predict_net)

##################################################################################
# Add accuracy and optionally print prototxt for model we just created

# Add an accuracy feature to the model for convenient reporting
accuracy = brew.accuracy(test_model, ['softmax', 'label' ], 'accuracy')

# Print the protoxt for the model (optional)
print "#################################"
print str(test_model.net.Proto())
#with open("test_model_predict_net_MINE.pbtxt", 'w') as fid:
#    fid.write(str(test_model.net.Proto()))
#

##################################################################################
# Run the test

# Run a test pass on the test net (same as in MNIST tutorial)
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
avg = 0.0

test_iters = 100

for i in range(test_iters):
	workspace.RunNet(test_model.net)
	acc = workspace.FetchBlob('accuracy')
	print "Batch: ",i," Accuracy: ",acc
	avg += acc

print "*********************************************"
print "Final Test Accuracy: ",avg/test_iters






