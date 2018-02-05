# NAI
# This is the basic MNIST tutorial from the caffe2 documentation
# 	trimmed down to only what is needed for training and saving the pb file.
#   The original tutorial has been decomposed into a single sequential program
#   for better understanding.

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
predict_net_out = "mnist_predict_net.pb" # Note: these are in PWD
init_net_out = "mnist_init_net.pb"
training_iters = 500

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

for i,op in enumerate(train_model.net.Proto().op):
    print "\n******************************"
    print "OP: ", i
    print "******************************"
    print "OP_NAME: ",op.name
    print "OP_TYPE: ",op.type
    print "OP_INPUT: ",op.input
    print "OP_OUTPUT: ",op.output

print train_model.GetOptimizationParamInfo()
print "\n***************** PRINTING MODEL PARAMS *******************"
for param in train_model.params:
    print "PARAM: ",param

print train_model.params

#exit()
'''
# At this point, the basic architecture of the model exists but none of required operators for training exist.
# Calculate the cross entropy of the label array with the softmax array
xent = train_model.LabelCrossEntropy(['softmax','label'], 'xent')
# compute the expected loss
loss = train_model.AveragedLoss(xent,"loss")
# track the accuracy of the model by adding the accuracy operator to the model
# this is a bookkeeping operator and is not required for operation
accuracy = brew.accuracy(train_model, ['softmax','label'], 'accuracy')
# *** KEY: add the gradient operators to the model ***
# use the average loss to add gradient operators to the model
# gradient is computed with respect to the loss which was just computed
train_model.AddGradientOperators(['loss'])
# BEGIN SGD ALGO...
# do a simple stochastic gradient descent
# iter op is a counter for the number of iterations run in training
ITER = brew.iter(train_model, "iter")
# set the learning rate schedule to lr = base_lr * (t^gamma)
LR = train_model.LearningRate(ITER,"LR", base_lr=-0.1, policy="step", stepsize=1, gamma=0.999)
# Define the constant ONE which will be used in the gradient update. Since this never needs to be created
# again, we can explicitly place it in the param_init_net
ONE = train_model.param_init_net.ConstantFill([],"ONE", shape=[1], value=1.0)
# Now, for each parameter in the model, do the gradinet updates
# The update is a simple weighted sum: param = param + param_grad*LR
for param in train_model.params:
	# Note, we used the model helper class to easily access the params of the model
	param_grad = train_model.param_to_grad[param]
	# Element-wise weighted sum of several data, weight tensor pairs.
	# Input should be in the form X_0, weight_0, X_1, weight_1, ... where X_i all
	# have the same shape, and weight_i are size 1 tensors that specifies the weight
	# of each vector
	# param = param + param_grad*LR
	train_model.WeightedSum([param, ONE, param_grad, LR], param)
'''

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

##################################################################################
#### Save the trained model for testing later

# save as two protobuf files (predict_net.pb and init_net.pb)
# predict_net.pb defines the architecture of the network
# init_net.pb defines the network params/weights
print "Saving the trained model to predict/init.pb files"
deploy_model = model_helper.ModelHelper(name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())

print "Done, exiting..."
