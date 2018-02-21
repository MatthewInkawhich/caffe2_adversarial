# NAI

# This script allows us to resume training the stacked optical flow network that was
# 	created with the CNNM_oflowstack_train_and_save.py

# import dependencies
print "Import Dependencies..."
from matplotlib import pyplot
import numpy as np 
import os
import glob
import shutil
import random
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import cv2
import JesterDatasetHandler as jdh


##################################################################################
# Gather Inputs
train_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TrainDictionary_5class.txt")
INIT_NET = os.path.join(os.path.expanduser('~'),"DukeML/caffe2_sandbox/Temporal_Networks/CNNM_1epoch_jester_init_net.pb")
init_net_out = "CNNM_3epoch_jester_init_net.pb"
checkpoint_iters = 1000
num_epochs = 2


gpu_no = 0
device_opts = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)


##################################################################################
# MAIN

print "Entering Main..."


##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW", "gpu_id": gpu_no, "use_cudnn": True}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="CNNM_jester_train", init_params=False, arg_scope=arg_scope)
train_model.param_init_net.RunAllOnGPU()
train_model.net.RunAllOnGPU()


##################################################################################
#### Add the model definition the the model object

# CNN-M-2048 [https://arxiv.org/pdf/1405.3531.pdf]
# As mentioned in two stream paper, we omit the norm layer in conv2

# O = ( (W - K + 2P) / S ) + 1

def Add_CNN_M(model,data):

	# Shape here = 20x100x100
	with core.DeviceScope(device_opts):
		##### CONV-1
		conv1 = brew.conv(model, data, 'conv1', dim_in=20, dim_out=96, kernel=7, stride=2, pad=0)
		#norm1 = brew.lrn(model, conv1, 'norm1', order="NCHW")
		# Shape here = 96x47x47
		pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
		# Shape here = 96x23x23
		relu1 = brew.relu(model, pool1, 'relu1')

		# Shape here = 96x23x23

		##### CONV-2
		conv2 = brew.conv(model, 'relu1', 'conv2', dim_in=96, dim_out=256, kernel=5, stride=2, pad=1)
		# Shape here = 256x11x11
		pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
		# Shape here = 256x5x5
		relu2 = brew.relu(model, pool2, 'relu2')

		# Shape here = 256x5x5

		##### CONV-3
		conv3 = brew.conv(model, 'relu2', 'conv3', dim_in=256, dim_out=512, kernel=3, stride=1, pad=1)
		# Shape here = 512x5x5
		relu3 = brew.relu(model, conv3, 'relu3')

		# Shape here = 512x5x5

		##### CONV-4
		conv4 = brew.conv(model, 'relu3', 'conv4', dim_in=512, dim_out=512, kernel=3, stride=1, pad=1)
		# Shape here = 512x5x5
		relu4 = brew.relu(model, conv4, 'relu4')

		# Shape here = 512x5x5

		##### CONV-5
		conv5 = brew.conv(model, 'relu4', 'conv5', dim_in=512, dim_out=512, kernel=3, stride=1, pad=1)
		# Shape here = 512x5x5
		pool5 = brew.max_pool(model, conv5, 'pool5', kernel=2, stride=2)
		# Shape here = 512x2x2
		relu5 = brew.relu(model, pool5, 'relu5')

		# Shape here = 512x2x2

		fc6 = brew.fc(model, relu5, 'fc6', dim_in=512*2*2, dim_out=4096)
		relu6 = brew.relu(model, fc6, 'relu6')

		# Shape here = 1x4096

		fc7 = brew.fc(model, relu6, 'fc7', dim_in=4096, dim_out=4096)
		relu7 = brew.relu(model, fc7, 'relu7')

		# Shape here = 1x4096

		fc8 = brew.fc(model, relu7, 'fc8', dim_in=4096, dim_out=5)

		# Shape here = 1x5

		softmax = brew.softmax(model,fc8, 'softmax')

		return softmax


# Add the model definition to the model
softmax=Add_CNN_M(train_model, 'data')


# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
#train_model.param_init_net = train_model.param_init_net.AppendNet(tmp_param_net)
train_model.param_init_net = tmp_param_net

##################################################################################
#### Step 3: Add training operators to the model

ITER = brew.iter(train_model, "iter")
train_model.Checkpoint([ITER] + train_model.params, [], db="cnnm_checkpoint1_%05d.lmdb", db_type="lmdb", every=checkpoint_iters)


xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])

optimizer.build_sgd(train_model,base_learning_rate=0.01, policy="step", stepsize=10000, gamma=0.1, momentum=0.9)

##################################################################################
#### Run the training procedure

# Initialization.

train_dataset = jdh.Jester_Dataset(dictionary_file=train_dictionary,seq_size=10)

# Prime the workspace with some data so we can run init net once
for image, label in train_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# run the param init network once
workspace.RunNetOnce(train_model.param_init_net)
# create the network
workspace.CreateNet(train_model.net, overwrite=True)


# Set the total number of iterations and track the accuracy and loss
accuracy = []
loss = []

batch_size = 50
# Manually run the network for the specified amount of iterations
for epoch in range(num_epochs):

	for index, (image, label) in enumerate(train_dataset.read(batch_size)):
		workspace.FeedBlob("data", image, device_option=device_opts)
		workspace.FeedBlob("label", label, device_option=device_opts)
		workspace.RunNet(train_model.net) # SEGFAULT HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		curr_acc = workspace.FetchBlob('accuracy')
		curr_loss = workspace.FetchBlob('loss')
		accuracy.append(curr_acc)
		loss.append(curr_loss)
		print "[{}][{}/{}] loss={}, accuracy={}".format(epoch, index, int(len(train_dataset) / batch_size),curr_loss, curr_acc)


##################################################################################
#### Save the trained model for testing later

# save as two protobuf files (predict_net.pb and init_net.pb)
# predict_net.pb defines the architecture of the network
# init_net.pb defines the network params/weights
print "Saving the trained model to predict/init.pb files..."
deploy_model = model_helper.ModelHelper(name="cnnm_deploy", arg_scope=arg_scope, init_params=False)
Add_CNN_M(deploy_model, "data")

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())

print "Done, saving..."

# After execution is done lets plot the values
pyplot.plot(np.array(loss),'b', label='loss')
pyplot.plot(np.array(accuracy),'r', label='accuracy')
pyplot.legend(loc='upper right')
pyplot.show()



