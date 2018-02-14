import numpy as np
import os
import shutil
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter


########################################################################
# Helper functions
########################################################################

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


def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = brew.accuracy(model, [softmax, label], "accuracy")
    return accuracy


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


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2*pad)//stride) + 1
    new_width = ((width - kernel + 2*pad)//stride) + 1
    return new_height, new_width








########################################################################
# Model definitions
########################################################################

# Basic MNIST LeNet model from tutorial
def AddLeNetModel(model, data, num_classes, image_height, image_width, image_channels):
    # Image size: 28 x 28 -> 24 x 24
    ############# Change dim_in value if images are more than 1 color channel
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    h,w = update_dims(height=image_height, width=image_width, kernel=5, stride=1, pad=0)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    h,w = update_dims(height=h, width=w, kernel=2, stride=2, pad=0)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    h,w = update_dims(height=h, width=w, kernel=5, stride=1, pad=0)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    h,w = update_dims(height=h, width=w, kernel=2, stride=2, pad=0)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    ############# Change dim_in value if images are not 28x28
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * h * h, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, num_classes)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax



def AddUpgradedLeNetModel(model, data, num_classes, image_height, image_width, image_channels):
    # Image size: 64x64
    conv1 = brew.conv(model, data, 'conv1', dim_in=image_channels, dim_out=32, kernel=5)
    h,w = update_dims(height=image_height, width=image_width, kernel=5, stride=1, pad=0)
    # Image size: 60x60
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    h,w = update_dims(height=h, width=w, kernel=2, stride=2, pad=0)
    relu1 = brew.relu(model, pool1, 'relu1')
    # Image size: 30x30
    conv2 = brew.conv(model, relu1, 'conv2', dim_in=32, dim_out=64, kernel=5)
    h,w = update_dims(height=h, width=w, kernel=5, stride=1, pad=0)
    # Image size: 26x26
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    h,w = update_dims(height=h, width=w, kernel=2, stride=2, pad=0)
    relu2 = brew.relu(model, pool2, 'relu2')
    # Image size: 13x13
    conv3 = brew.conv(model, relu2, 'conv3', dim_in=64, dim_out=64, kernel=5)
    h,w = update_dims(height=h, width=w, kernel=5, stride=1, pad=0)
    # Image size: 9x9
    pool3 = brew.max_pool(model, conv3, 'pool3', kernel=2, stride=2)
    h,w = update_dims(height=h, width=w, kernel=2, stride=2, pad=0)
    relu3 = brew.relu(model, pool3, 'relu3')
    # Image size: 4x4
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    ############# Change dim_in value if images are not 28x28
    fc3 = brew.fc(model, relu3, 'fc3', dim_in=64 * h * w, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, num_classes)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax
