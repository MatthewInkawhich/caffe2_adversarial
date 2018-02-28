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
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

import cv2
import JesterDatasetHandler as jdh
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'lib'))
import model_defs


##################################################################################
# Gather Inputs
train_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TrainDictionary_5class.txt")
INIT_NET = os.path.join(os.path.expanduser('~'),"DukeML/caffe2_sandbox/Temporal_Networks/CNNM_10epoch_jester_init_net.pb")
saved_checkpoint = os.path.join(os.path.expanduser('~'), 'DukeML', 'models', 'CNNM', 'cnnm_checkpoint_16000.lmdb')
PREDICT_NET = os.path.join(os.path.expanduser('~'),"DukeML/caffe2_sandbox/Temporal_Networks/CNNM_10_jester_predict_net.pb")
init_net_out = "CNNM_10epoch_jester_init_net.pb"
checkpoint_iters = 1000
num_epochs = 7

gpu_no = 0
device_opts_gpu = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)
device_opts_cpu = core.DeviceOption(caffe2_pb2.CPU, 0)


##################################################################################
# MAIN

print "Entering Main..."


##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW"}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="CNNM_jester_train", init_params=False, arg_scope=arg_scope)

# Populate the model obj with the predict net def
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
    predict_net_proto.device_option.CopyFrom(device_opts_cpu)
tmp_predict_net = core.Net(predict_net_proto)
train_model.net = train_model.net.AppendNet(tmp_predict_net)


# Load params and blobs from checkpoint lmdb
#workspace.RunOperatorOnce(
#      core.CreateOperator("Load", [], [], absolute_path=1, db=saved_checkpoint, db_type="lmdb", keep_device=1, load_all=1))


init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
    init_net_proto.device_option.CopyFrom(device_opts_cpu)
tmp_param_net = core.Net(init_net_proto)
train_model.param_init_net = train_model.param_init_net.AppendNet(tmp_param_net)

for i,op in enumerate(init_net_proto.op):
	param_name = op.output[0]
	if param_name != 'data':
		assert(op.arg[0].name == "shape")
		tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
		train_model.create_param(param_name=op.output[0], shape=op.arg[0].ints, initializer=initializers.ExternalInitializer(), tags=tags)

print(train_model.net.Proto())

exit()







train_model.param_init_net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)
train_model.net.RunAllOnGPU(gpu_id=gpu_no, use_cudnn=True)

##################################################################################



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
		workspace.FeedBlob("data", image, device_option=device_opts_gpu)
		workspace.FeedBlob("label", label, device_option=device_opts_gpu)
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
print "Saving the trained model to predict/init.pb files (GPU)..."
deploy_model_gpu = model_helper.ModelHelper(name="cnnm_deploy_gpu", arg_scope=arg_scope, init_params=False)
model_defs.Add_CNN_M(deploy_model_gpu, "data", device_opts_gpu)

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model_gpu.param_init_net)
workspace.CreateNet(deploy_model_gpu.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model_gpu.net, deploy_model_gpu.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())



print "Saving the trained model to predict/init.pb files (CPU)..."
deploy_model_cpu = model_helper.ModelHelper(name="cnnm_deploy_cpu", arg_scope=arg_scope, init_params=False)
model_defs.Add_CNN_M(deploy_model_cpu, "data", device_opts_cpu)

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model_cpu.param_init_net)
workspace.CreateNet(deploy_model_cpu.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model_cpu.net, deploy_model_cpu.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())


print "Done, saving..."

# After execution is done lets plot the values
#pyplot.plot(np.array(loss),'b', label='loss')
#pyplot.plot(np.array(accuracy),'r', label='accuracy')
#pyplot.legend(loc='upper right')
#pyplot.show()



