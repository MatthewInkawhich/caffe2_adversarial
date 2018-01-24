from __future__ import print_function
from matplotlib import pyplot
import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer

# data and root paths
### CHANGE THESE
current_folder = os.path.join(os.path.expanduser('~'), 'caffe2_examples')
data_folder = os.path.join(os.path.expanduser('~'), 'MSTAR_images')
root_folder = os.path.join(current_folder, 'mstar')
if not os.path.isdir(root_folder):
    os.makedirs(root_folder)

# resetting workspace sets root_folder as "root" dir (bookkeeping files go here)
workspace.ResetWorkspace(root_folder)
print("training data folder:" + data_folder)
print("workspace root folder:" + root_folder)

num_classes = 8
training_lmdb = os.path.join(data_folder, 'shuffled_train_lmdb')
validation_lmdb = os.path.join(data_folder, 'shuffled_validate_lmdb')
testing_lmdb = os.path.join(data_folder, 'shuffled_test_lmdb')
training_net_batch_size = 32
validation_net_batch_size = 452
testing_net_batch_size = 442
training_iters = 300
validation_interval = 20



########################################################################
# Functions
########################################################################
# loads data from DB
def AddInput(model, batch_size, db, db_type):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type=db_type)
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label


# define model architecture in terms of layers
def AddLeNetModel(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.

    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''

    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 29 * 29, dim_out=500)
    #fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, num_classes)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax


# accuracy operator
def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


# adds training operators to model
def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    AddAccuracy(model, softmax, label)
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    opt = optimizer.build_sgd(model, base_learning_rate=0.1)
    for param in model.GetOptimizationParamInfo():
        opt(model.net, model.param_init_net, param)


# collects stats to inspect later
def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.



########################################################################
# Define training, testing, and deployment models
########################################################################
arg_scope = {"order": "NCHW"}
# Training model
train_model = model_helper.ModelHelper(
    name="mstar_train", arg_scope=arg_scope)
data, label = AddInput(
    train_model, batch_size=training_net_batch_size,
    db=training_lmdb,
    db_type='lmdb')
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)
AddBookkeepingOperators(train_model)

# Validation model
val_model = model_helper.ModelHelper(
    name="mstar_val", arg_scope=arg_scope, init_params=False)
data, label = AddInput(
    val_model, batch_size=validation_net_batch_size,
    db=validation_lmdb,
    db_type='lmdb')
softmax = AddLeNetModel(val_model, data)
AddAccuracy(val_model, softmax, label)

# Testing model. We will set the batch size to 100, so that the testing
# pass is 100 iterations (10,000 images in total).
# For the testing model, we need the data input part, the main LeNetModel
# part, and an accuracy part. Note that init_params is set False because
# we will be using the parameters obtained from the train model.
test_model = model_helper.ModelHelper(
    name="mstar_test", arg_scope=arg_scope, init_params=False)
data, label = AddInput(
    test_model, batch_size=testing_net_batch_size,
    db=testing_lmdb,
    db_type='lmdb')
softmax = AddLeNetModel(test_model, data)
AddAccuracy(test_model, softmax, label)

# Deployment model. We simply need the main LeNetModel part.
deploy_model = model_helper.ModelHelper(
    name="mstar_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")
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
_ = visualize.NCHW.ShowSingle(data[0][0])
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
