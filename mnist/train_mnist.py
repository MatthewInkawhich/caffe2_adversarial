# MatthewInkawhich
# Use this script to train a simple MNIST net on the lmdbs of your choice and save the trained model
# LINES TO MODIFY:
#   - Must manually set Configs
#   - Modify AddLeNetModel function if your images are:
#       (a) not 1 color channel
#       (b) not 28x28

from __future__ import print_function
from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'lib'))
import model_defs


########################################################################
# Configs
########################################################################
root_folder = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mnist_output') #where bookkeeping files are outputted
save_trained_model_loc = root_folder
init_net_out = os.path.join(save_trained_model_loc, 'mnist_init_net.pb')
predict_net_out = os.path.join(save_trained_model_loc, 'mnist_predict_net.pb')
training_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'custom_mnist', 'train_lmdb')
validation_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'custom_mnist', 'validate_lmdb')
testing_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'custom_mnist', 'test_lmdb')
num_classes = 10                    #number of image classes
training_net_batch_size = 50        #batch size for training
validation_images = 2000            #total number of validation images
testing_images = 2000               #total number of testing images
training_iters = 500               #training iterations
validation_interval = 50            #validate every ... training iterations
image_height = 28
image_width = 28
image_channels = 1

########################################################################
# create root_folder if not already there
if not os.path.isdir(root_folder):
    os.makedirs(root_folder)

# resetting workspace sets root_folder as "root" dir (bookkeeping files go here)
workspace.ResetWorkspace(root_folder)
print("workspace output folder:" + root_folder)


########################################################################
# Functions
########################################################################

# adds training operators to model
def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    model_defs.AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    optimizer.build_sgd(
        model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=1,
        gamma=0.999,
    )



########################################################################
# Define training, testing, and deployment models
########################################################################
arg_scope = {"order": "NCHW"}
# Training model
train_model = model_helper.ModelHelper(
    name="train_net", arg_scope=arg_scope)
data, label = model_defs.AddInput(
    train_model, batch_size=training_net_batch_size,
    db=training_lmdb,
    db_type='lmdb')
softmax = model_defs.AddLeNetModel(train_model, data, num_classes, image_height, image_width, image_channels)
AddTrainingOperators(train_model, softmax, label)
model_defs.AddBookkeepingOperators(train_model)


# Validation model
val_model = model_helper.ModelHelper(
    name="val_net", arg_scope=arg_scope, init_params=False)
data, label = model_defs.AddInput(
    val_model, batch_size=validation_images,
    db=validation_lmdb,
    db_type='lmdb')
softmax = model_defs.AddLeNetModel(val_model, data, num_classes, image_height, image_width, image_channels)
model_defs.AddAccuracy(val_model, softmax, label)


# Testing model
test_model = model_helper.ModelHelper(
    name="test_net", arg_scope=arg_scope, init_params=False)
data, label = model_defs.AddInput(
    test_model, batch_size=testing_images,
    db=testing_lmdb,
    db_type='lmdb')
softmax = model_defs.AddLeNetModel(test_model, data, num_classes, image_height, image_width, image_channels)
model_defs.AddAccuracy(test_model, softmax, label)

print("###### Hello!!")
print("Params and grads:")
for param, grad in test_model.param_to_grad.items():
    print(str(param) + ', ' + str(grad))

# Deployment model
deploy_model = model_helper.ModelHelper(
    name="mstar_deploy", arg_scope=arg_scope, init_params=False)
model_defs.AddLeNetModel(deploy_model, "data", num_classes, image_height, image_width, image_channels)
# You may wonder what happens with the param_init_net part of the deploy_model.
# No, we will not use them, since during deployment time we will not randomly
# initialize the parameters, but load the parameters from the db.

# Dump all protobufs to disk for later inspection
with open(os.path.join(root_folder, "train_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.net.Proto()))
with open(os.path.join(root_folder, "train_init_net.pbtxt"), 'w') as fid:
    fid.write(str(train_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "test_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.net.Proto()))
with open(os.path.join(root_folder, "test_init_net.pbtxt"), 'w') as fid:
    fid.write(str(test_model.param_init_net.Proto()))
with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
    fid.write(str(deploy_model.net.Proto()))
print("Protocol buffers files have been created in your root folder: " + root_folder)



########################################################################
# Run training procedure
########################################################################
# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
# creating the network
workspace.CreateNet(train_model.net, overwrite=True)
# initialize and create validation network
workspace.RunNetOnce(val_model.param_init_net)
workspace.CreateNet(val_model.net, overwrite=True)
# variables to track the accuracy & loss
accuracy = np.zeros(training_iters)
loss = np.zeros(training_iters)
# Now, we will manually run the network for 200 iterations.
for i in range(training_iters):
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    if (i % validation_interval == 0):
        print("Training iter: ", i)
        #run validation
        workspace.RunNet(val_model.net.Proto().name)
        val_accuracy = workspace.FetchBlob('accuracy')
        print("Validation accuracy: ", str(val_accuracy))


# After the execution is done, let's plot the values.
pyplot.plot(loss, 'b')
pyplot.plot(accuracy, 'r')
pyplot.legend(('Loss', 'Accuracy'), loc='upper right')
pyplot.show()



########################################################################
# View data
########################################################################
pyplot.figure()
data = workspace.FetchBlob('data')
#_ = visualize.NCHW.ShowSingle(data[0][0])
_ = visualize.NCHW.ShowMultiple(data)
pyplot.figure()
softmax = workspace.FetchBlob('softmax')
_ = pyplot.plot(softmax[0], 'ro')
pyplot.title('Prediction for the first image')
pyplot.show()



########################################################################
# Run test on test net
########################################################################
#run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
workspace.RunNet(test_model.net.Proto().name)
test_accuracy = workspace.FetchBlob('accuracy')
# After the execution is done, let's print the mean accuracy value.
print('\ntest_accuracy: %f' % test_accuracy)



########################################################################
# Save deploy model with trained weights and biases
########################################################################
# print("\n\nSaving the trained model to " + save_trained_model_loc)
# # Use the MOBILE EXPORTER to save the deploy model as pbs
# # https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
# workspace.RunNetOnce(deploy_model.param_init_net)
# workspace.CreateNet(deploy_model.net, overwrite=True)
# init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
# with open(init_net_out, 'wb') as f:
#     f.write(init_net.SerializeToString())
# with open(predict_net_out, 'wb') as f:
#     f.write(predict_net.SerializeToString())
