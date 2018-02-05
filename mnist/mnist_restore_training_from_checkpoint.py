# NAI
# This is meant to show how we periodically save checkpoints during training.
# This script will go with another script that shows how to restore training from
#  	a saved checkpoint.

# import dependencies
print "Import Dependencies..."
from matplotlib import pyplot
import numpy as np 
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter

##################################################################################
# MAIN

print "Entering Main..."

##################################################################################
# Gather Inputs
train_lmdb = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnist-lmdb/mnist-train-nchw-lmdb")
saved_checkpoint = os.path.join(os.path.expanduser('~'),"DukeML/caffe2_sandbox/mnist/mnist_lenet_checkpoint_00005.lmdb")
training_iters = 10
checkpoint_iters = 5

# Make sure the training lmdb exists
if not os.path.exists(train_lmdb):
	print "ERROR: train lmdb NOT found"
	exit()

##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW"}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)

# Load the model from the checkpoint into train_model
#loaded_model = brew.Load(train_model, 'loaded_model', db=saved_checkpoint, db_type="lmdb",  load_all=1)
train_model.Load(db=saved_checkpoint, db_type="lmdb",  load_all=1)

exit()





##################################################################################
#### Step 1: Add Input Data

# Go read the data from the lmdb and format it properly
# Since the images are stored as 8-bit ints, we will read them as such
# We are using the TensorProtosDBInput because the lmdbs were created with a TensorProtos object
data_uint8, label = train_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=64, db=train_lmdb, db_type='lmdb')
# cast the 8-bit data to floats
data = train_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
# scale data from [0,255] -> [0,1]
data = train_model.Scale(data,data,scale=float(1./256))
# enforce a stopgradient because we do not need the gradient of the data for the backward pass
data = train_model.StopGradient(data,data)
# Here, we have access to the data and the labels
# the data is of shape [batch_size,num_channels,width,height]
# for mnist the shape is [bsize, 1, 28, 28]

##################################################################################
#### Step 2: Add the model definition the the model object
# This is where we specify the network architecture from 'conv1' -> 'softmax'
# [Arch: data->conv->pool->...->fc->relu->softmax]

def AddLeNetModel(model, data):
	conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
	pool1 = brew.max_pool(model, conv1, 'pool1',kernel=2,stride=2)
	conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
	pool2 = brew.max_pool(model, conv2, 'pool2',kernel=2, stride=2)
	fc3 = brew.fc(model, pool2, 'fc3', dim_in=50*4*4, dim_out=500)
	fc3 = brew.relu(model, fc3, fc3)
	pred = brew.fc(model,fc3,'pred',500,10)
	softmax = brew.softmax(model,pred, 'softmax')

softmax=AddLeNetModel(train_model, data)

##################################################################################
#### Step 3: Add training operators to the model
# TODO: use the optimizer class here instead of doing sgd by hand

xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])
opt = optimizer.build_sgd(train_model, base_learning_rate=0.1)
for param in train_model.GetOptimizationParamInfo():
    opt(train_model.net, train_model.param_init_net, param)



#model.Checkpoint([ITER] + model.params, [], db="mnist_lenet_checkpoint_%05d.lmdb", db_type="lmdb", every=20)
ITER = brew.iter(train_model, "iter")
train_model.Checkpoint([ITER] + train_model.params, [], db="mnist_lenet_checkpoint_%05d.lmdb", db_type="lmdb", every=checkpoint_iters)

##################################################################################
#### Run the training procedure

# run the param init network once
workspace.RunNetOnce(train_model.param_init_net)
# create the network
workspace.CreateNet(train_model.net, overwrite=True)
# Set the total number of iterations and track the accuracy and loss
total_iters = training_iters
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)
# Manually run the network for the specified amount of iterations
for i in range(total_iters):
	workspace.RunNet(train_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "Iter: {}, loss: {}, accuracy: {}".format(i, loss[i], accuracy[i])

# After execution is done lets plot the values
pyplot.plot(loss,'b', label='loss')
pyplot.plot(accuracy,'r', label='accuracy')
pyplot.legend(loc='upper right')
pyplot.show()

print "Done, exiting..."





