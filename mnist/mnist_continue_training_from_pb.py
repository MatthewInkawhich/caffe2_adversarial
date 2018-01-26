# NAI
# This script gives an example of how to continue training from a
#	saved pb file. The inputs to this script are the train lmdb,
#	the predict net pb file, and the init net pb file. The output is
#	a new set of predict/init net pb files.

# Methodology:
#	- create new model helper
#	- add data layer with train lmdb
#	- add predict net from input predict_net.pb
#	- add init net from input init_net.pb
#	- add the training operators
# 	- run the training
#	- save the new model

# Problem - cant get it to improve the accuracy !!!
# It doesnt seem like it is learning anything anymore

import numpy as np
import matplotlib.pyplot as plt
import os
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter

##################################################################################
# Get inputs

TRAIN_LMDB = os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/mnist-lmdb/mnist-train-nchw-lmdb")
INIT_NET = "mnist_init_net.pb"
PREDICT_NET = "mnist_predict_net.pb"
predict_net_out = "UPDATED_mnist_predict_net.pb" # Note: these are in PWD
init_net_out = "UPDATED_mnist_init_net.pb"
train_iters = 100

# Make sure the specified inputs exist
if ((not os.path.exists(TRAIN_LMDB)) or (not os.path.exists(INIT_NET)) or (not os.path.exists(PREDICT_NET))):
	print "ERROR: An input was not found"
	exit()

##################################################################################
# Create a model object (using model helper)

arg_scope = {"order": "NCHW"}
train_model = model_helper.ModelHelper(name="train_model", arg_scope=arg_scope, init_params=False)


##################################################################################
# Add the input ('data') stuff to the model

data_uint8, label = train_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=100, db=TRAIN_LMDB, db_type='lmdb')
data = train_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
data = train_model.Scale(data,data,scale=float(1./256))
data = train_model.StopGradient(data,data)

##################################################################################
# Add the pretrained model stuff (info from the pbs) to the model helper object

# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
train_model.param_init_net = train_model.param_init_net.AppendNet(tmp_param_net)

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
train_model.net = train_model.net.AppendNet(tmp_predict_net)



# Print some info about the ops contained in the init_net pb
# This is where we extract the names of the params that must be added to the model
#	for optimization and where we get the shape of the data for initialization
print "\n***************** OPS IN INIT NET PROTO *******************"
for i,op in enumerate(init_net_proto.op):
	print "\n******************************"
	print "OP: ", i
	print "******************************"
	print "OP_NAME: ",op.name
	print "OP_TYPE: ",op.type
	print "OP_INPUT: ",op.input 
	print "OP_OUTPUT: ",op.output
	print "OP_OUTPUT[0]: ",op.output[0]
	print "ATTRS: ", op.arg[0]
	print type(op.arg[0])
	print op.arg[0].ints


##################################################################################
# Add params to the network so we can train

# The required params are in the init net proto. We must extract their names and 
#	shapes and set them to be externally initialized. We must finally 'create' the 
# 	param so the optimizer can operate on it
print "\n***************** INITIALIZING MODEL PARAMS *******************"
# Want to add all of the op.output[0]'s (except data) as params and initialize them externally
for i,op in enumerate(init_net_proto.op):

	param_name = op.output[0]

	if param_name != 'data':
		assert(op.arg[0].name == "shape")
		print "initializing: ", param_name
		tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
		train_model.create_param(param_name=op.output[0], shape=op.arg[0].ints, initializer=initializers.ExternalInitializer(), tags=tags)


train_model.GetAllParams()
print "\n***************** PRINTING MODEL PARAMS *******************"
for param in train_model.params:
	print "PARAM: ",param

#exit()



##################################################################################
# Add the training operators to the model

xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])
opt = optimizer.build_sgd(train_model, base_learning_rate=0.1)
for param in train_model.GetOptimizationParamInfo():
    opt(train_model.net, train_model.param_init_net, param)



''' OLD WAY OF SPECIFYING SGD (NOW WE JUST USE THE OPTIMIZER CLASS)
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

'''
# Params that will be optimized during training
print "\n***************** OPT PARAM INFO *******************"
print train_model.GetOptimizationParamInfo()
exit()
'''

#print "GOA: ",train_model.gradient_ops_added

##################################################################################
# Run the training

workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)

total_iters = train_iters
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
	workspace.RunNet(train_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "accuracy: ", accuracy[i]
	print "loss: ", loss[i]

plt.plot(loss, 'b', label="loss")
plt.plot(accuracy, 'r', label="accuracy")
plt.legend(loc="upper right")
plt.xlabel("Iteration")
plt.ylabel("Loss and Accuracy")
plt.title("Loss and Accuracy through training")
plt.show()

exit()

##################################################################################
# Save the retrained model with new pb files

print "Saving the new model to predict/init.pb files"
deploy_model = model_helper.ModelHelper(name="UPDATED_mnist_deploy", arg_scope=arg_scope, init_params=False)
deploy_model.net = core.Net(predict_net_proto)

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









