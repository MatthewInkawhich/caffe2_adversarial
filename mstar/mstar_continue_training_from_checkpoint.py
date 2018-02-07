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
training_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'train_lmdb')
saved_checkpoint = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_output', 'mnist_lenet_checkpoint00200.lmdb')
training_iters = 100
#checkpoint_iters = 5

# Make sure the training lmdb exists
if not os.path.exists(training_lmdb):
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
#workspace.Load(db=saved_checkpoint, db_type="lmdb", load_all=1)

exit()
