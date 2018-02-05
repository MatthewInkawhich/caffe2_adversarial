# NAI

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
train_lmdb = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11-lmdb/UCF11-test-lmdb")
predict_net_out = "vgg_predict_net.pb" # Note: these are in PWD
init_net_out = "vgg_init_net.pb"
training_iters = 100

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
train_model = model_helper.ModelHelper(name="vgg_train", arg_scope=arg_scope)

##################################################################################
#### Step 1: Add Input Data

# Go read the data from the lmdb and format it properly
# Since the images are stored as 8-bit ints, we will read them as such
# We are using the TensorProtosDBInput because the lmdbs were created with a TensorProtos object
data_uint8, label = train_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=30, db=train_lmdb, db_type='lmdb')
# cast the 8-bit data to floats
data = train_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
# scale data from [0,255] -> [0,1]
data = train_model.Scale(data,data,scale=float(1./256))
# enforce a stopgradient because we do not need the gradient of the data for the backward pass
data = train_model.StopGradient(data,data)


##################################################################################
#### Step 2: Add the model definition the the model object
# This is where we specify the network architecture from 'conv1' -> 'softmax'

# My implementation of VGG-D model
# Inspired by: http://vast.uccs.edu/~adhamija/blog/VGG_multiGPU.html
def nate_VGG_d(model,data):

	# (?) - what is default weight fillers of these layers
	# (?) - should i add dropout to the fc layers?
	# (*) - make sure the output dimension is kosher

	# Shape here = 3x224x224

	# Conv group
	conv1_1 = brew.conv(model, data, 'conv1_1', dim_in=3, dim_out=64, kernel=3, pad=1)
	relu1_1 = brew.relu(model, conv1_1, 'relu1_1')
	conv1_2 = brew.conv(model, relu1_1, 'conv1_2', dim_in=64, dim_out=64, kernel=3, pad=1)
	relu1_2 = brew.relu(model, conv1_2, 'relu1_2')

	# Shape here = 64x224x224

	# Max Pool
	pool1 = brew.max_pool(model, relu1_2, 'pool1', kernel=2, stride=2)

	# Shape here = 64x112x112

	# Conv group
	conv2_1 = brew.conv(model, pool1, 'conv2_1', dim_in=64, dim_out=128, kernel=3, pad=1)
	relu2_1 = brew.relu(model, conv2_1, 'relu2_1')
	conv2_2 = brew.conv(model, relu2_1, 'conv2_2', dim_in=128, dim_out=128, kernel=3, pad=1)
	relu2_2 = brew.relu(model, conv2_2, 'relu2_2')

	# Shape here = 128x112x112

	# Max Pool
	pool2 = brew.max_pool(model, relu2_2, 'pool2', kernel=2, stride=2)

	# Shape here = 128x56x56

	# Conv group
	conv3_1 = brew.conv(model, pool2, 'conv3_1', dim_in=128, dim_out=256, kernel=3, pad=1)
	relu3_1 = brew.relu(model, conv3_1, 'relu3_1')
	conv3_2 = brew.conv(model, relu3_1, 'conv3_2', dim_in=256, dim_out=256, kernel=3, pad=1)
	relu3_2 = brew.relu(model, conv3_2, 'relu3_2')
	conv3_3 = brew.conv(model, relu3_2, 'conv3_3', dim_in=256, dim_out=256, kernel=3, pad=1)
	relu3_3 = brew.relu(model, conv3_3, 'relu3_3')

	# Shape here = 256x56x56

	# Max Pool
	pool3 = brew.max_pool(model, relu3_3, 'pool3', kernel=2, stride=2)

	# Shape here = 256x28x28

	# Conv group
	conv4_1 = brew.conv(model, pool3, 'conv4_1', dim_in=256, dim_out=512, kernel=3, pad=1)
	relu4_1 = brew.relu(model, conv4_1, 'relu4_1')
	conv4_2 = brew.conv(model, relu4_1, 'conv4_2', dim_in=512, dim_out=512, kernel=3, pad=1)
	relu4_2 = brew.relu(model, conv4_2, 'relu4_2')
	conv4_3 = brew.conv(model, relu4_2, 'conv4_3', dim_in=512, dim_out=512, kernel=3, pad=1)
	relu4_3 = brew.relu(model, conv4_3, 'relu4_3')

	# Shape here = 512x28x28

	# Max Pool
	pool4 = brew.max_pool(model, relu4_3, 'pool4', kernel=2, stride=2)

	# Shape here = 512x14x14

	# Conv group
	conv5_1 = brew.conv(model, pool4, 'conv5_1', dim_in=512, dim_out=512, kernel=3, pad=1)
	relu5_1 = brew.relu(model, conv5_1, 'relu5_1')
	conv5_2 = brew.conv(model, relu5_1, 'conv5_2', dim_in=512, dim_out=512, kernel=3, pad=1)
	relu5_2 = brew.relu(model, conv5_2, 'relu5_2')
	conv5_3 = brew.conv(model, relu5_2, 'conv5_3', dim_in=512, dim_out=512, kernel=3, pad=1)
	relu5_3 = brew.relu(model, conv5_3, 'relu5_3')

	# Shape here = 512x14x14

	# Max Pool
	pool5 = brew.max_pool(model, relu5_3, 'pool5', kernel=2, stride=2)

	# Shape here = 512x7x7

	fc1 = brew.fc(model, pool5, 'fc1', dim_in=512*7*7, dim_out=4096)
	relu1 = brew.relu(model, fc1, 'relu1')

	# Shape here = 1x4096

	fc2 = brew.fc(model, relu1, 'fc2', dim_in=4096, dim_out=4096)
	relu2 = brew.relu(model, fc2, 'relu2')

	# Shape here = 1x4096

	fc3 = brew.fc(model, relu2, 'fc3', dim_in=4096, dim_out=11)

	# Shape here = 1x11

	softmax = brew.softmax(model,fc3, 'softmax')

	return softmax

'''
# http://vast.uccs.edu/~adhamija/blog/VGG_multiGPU.html
# This is the D version as described in Table 1 of VGG paper (https://arxiv.org/pdf/1409.1556.pdf)
def AddVGGModel_D(model,data):

	#----- 3 x 224 x 224 --> 64 x 224 x 224 -----#
	conv1_1 = brew.conv(model, data, 'conv1_1', 3, 64, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu1_1 = brew.relu(model, conv1_1, 'relu1_1')
	#----- 64 x 224 x 224 --> 64 x 224 x 224 -----#
	conv1_2 = brew.conv(model, relu1_1, 'conv1_2', 64, 64, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu1_2 = brew.relu(model, conv1_2, 'relu1_2')
	#----- 64 x 224 x 224 --> 64 x 112 x 112 -----#
	pool1 = brew.max_pool(model, relu1_2, 'pool1', kernel=2, stride=2)

	#----- 64 x 112 x 112 --> 128 x 112 x 112 -----#
	conv2_1 = brew.conv(model, pool1, 'conv2_1', 64, 128, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu2_1 = brew.relu(model, conv2_1, 'relu2_1')
	#----- 128 x 112 x 112 --> 128 x 112 x 112 -----#
	conv2_2 = brew.conv(model, relu2_1, 'conv2_2', 128, 128, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu2_2 = brew.relu(model, conv2_2, 'relu2_2')
	#----- 128 x 112 x 112 --> 128 x 56 x 56 -----#
	pool2 = brew.max_pool(model, relu2_2, 'pool2', kernel=2, stride=2)

	#----- 128 x 56 x 56 --> 256 x 56 x 56 -----#
	conv3_1 = brew.conv(model, pool2, 'conv3_1', 128, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu3_1 = brew.relu(model, conv3_1, 'relu3_1')
	#----- 256 x 56 x 56 --> 256 x 56 x 56 -----#
	conv3_2 = brew.conv(model, relu3_1, 'conv3_2', 256, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu3_2 = brew.relu(model, conv3_2, 'relu3_2')
	#----- 256 x 56 x 56 --> 256 x 56 x 56 -----#
	conv3_3 = brew.conv(model, relu3_2, 'conv3_3', 256, 256, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu3_3 = brew.relu(model, conv3_3, 'relu3_3')
	#----- 256 x 56 x 56 --> 256 x 28 x 28 -----#
	pool3 = brew.max_pool(model, relu3_3, 'pool3', kernel=2, stride=2)

	#----- 256 x 28 x 28 --> 512 x 28 x 28 -----#
	conv4_1 = brew.conv(model, pool3, 'conv4_1', 256, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu4_1 = brew.relu(model, conv4_1, 'relu4_1')
	#----- 512 x 28 x 28 --> 512 x 28 x 28 -----#
	conv4_2 = brew.conv(model, relu4_1, 'conv4_2', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu4_2 = brew.relu(model, conv4_2, 'relu4_2')
	#----- 512 x 28 x 28 --> 512 x 28 x 28 -----#
	conv4_3 = brew.conv(model, relu4_2, 'conv4_3', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu4_3 = brew.relu(model, conv4_3, 'relu4_3')
	#----- 512 x 28 x 28 --> 512 x 14 x 14 -----#
	pool4 = brew.max_pool(model, relu4_3, 'pool4', kernel=2, stride=2)

	#----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
	conv5_1 = brew.conv(model, pool4, 'conv5_1', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu5_1 = brew.relu(model, conv5_1, 'relu5_1')
	#----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
	conv5_2 = brew.conv(model, relu5_1, 'conv5_2', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu5_2 = brew.relu(model, conv5_2, 'relu5_2')
	#----- 512 x 14 x 14 --> 512 x 14 x 14 -----#
	conv5_3 = brew.conv(model, relu5_2, 'conv5_3', 512, 512, 3,pad=1,weight_init=('GaussianFill',{'mean':0.0, 'std':1e-2}))
	relu5_3 = brew.relu(model, conv5_3, 'relu5_3')
	#----- 512 x 14 x 14 --> 512 x 7 x 7 -----#
	pool5 = brew.max_pool(model, relu5_3, 'pool5', kernel=2, stride=2)

	fc6 = brew.fc(model, pool5, 'fc6', 25088, 4096)
	#        fc6 = brew.fc(model, pool5, 'fc6', 25088, 4096)
	relu6 = brew.relu(model, fc6,'relu6')

	drop6 = brew.dropout(model, relu6, 'drop6', ratio=0.5, is_test=0)

	fc7 = brew.fc(model, drop6, 'fc7', 4096, 4096)
	relu7 = brew.relu(model, fc7,'relu7')
	drop7 = brew.dropout(model, relu7,'drop7',ratio=0.5,is_test=0)

	fc8 = brew.fc(model, drop7, 'fc8', 4096, 11)
	#no_of_ids)
	softmax = brew.softmax(model, fc8, 'softmax')    
	
	return softmax
'''


# Add the model definition to the model
softmax=nate_VGG_d(train_model, data)

# Checkpoint here ?
#model.Checkpoint([ITER] + model.params, [], db="mnist_lenet_checkpoint_%05d.lmdb", db_type="lmdb", every=20)

##################################################################################
#### Step 3: Add training operators to the model
# TODO: use the optimizer class here instead of doing sgd by hand

xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])

optimizer.build_sgd(train_model,base_learning_rate=0.1, policy="step", stepsize=1, gamma=0.999)

'''
opt = optimizer.build_sgd(train_model, base_learning_rate=0.1)
for param in train_model.GetOptimizationParamInfo():
    opt(train_model.net, train_model.param_init_net, param)
'''


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
	workspace.RunNet(train_model.net) # SEGFAULT HERE
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





