# MatthewInkawhich
# Use this script to train a custom CNN without using LMDB for data layer input
# LINES TO MODIFY BEFORE RUNNING:
#   - Must manually set Configs


from __future__ import print_function
from matplotlib import pyplot
import numpy as np
import os
import shutil
import random
import skimage.io
import skimage.transform
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core, model_helper, net_drawer, workspace, visualize, brew, optimizer
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter

########################################################################
# Configs
########################################################################
root_folder = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_NO_LMDB_output') #where bookkeeping files are outputted
save_trained_model_loc = root_folder
init_net_out = os.path.join(save_trained_model_loc, 'mstar_NO_LMDB_init_net.pb')
predict_net_out = os.path.join(save_trained_model_loc, 'mstar_NO_LMDB_predict_net.pb')
#training_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'train_lmdb')
# validation_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'validate_lmdb')
# testing_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'test_lmdb')
TRAIN_DICTIONARY = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'training_labels')
VALIDATION_DICTIONARY = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'validation_labels')
TEST_DICTIONARY = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'testing_labels')
KEY_FILE = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar', 'MSTAR_KEYFILE')
num_classes = 8                   #number of image classes
training_net_batch_size = 30        #batch size for training
training_iters = 200               #training iterations
validation_images = 452            #total number of validation images
validation_interval = 25            #validate every ... training iterations
testing_images = 442               #total number of testing images
image_width = 64
image_height = 64
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
def rescale(img, input_height, input_width):
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        return skimage.transform.resize(img, (input_width, int(aspect * input_height)))
    elif aspect < 1:
        return skimage.transform.resize(img, (int(input_width/aspect), input_height))
    else:
        return skimage.transform.resize(img, (input_width, input_height))


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]


def prepare_image(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float(img)
    if (len(img.shape) < 3):
        img = np.expand_dims(img, axis=2)          # expand dims for greyscale images
    else:
        img = img[:, :, (2, 1, 0)]                 # RGB to BGR color order
    img = rescale(img, image_height, image_width)
    img = crop_center(img, image_height, image_width)
    img = img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    #img = img * 255 - 128                      # Subtract mean = 128
    return img.astype(np.float32)


def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]


def extract_categories(key_file):
    key_dict = {}
    f = open(key_file)
    for line in f:
        l = line.split()
        key_dict[l[0]] = l[1]

    print(key_dict)
    return key_dict


class MSTAR_Dataset(object):
    def __init__(self, dictionary_file):
        self.categories = extract_categories(KEY_FILE)
        self.image_files = [line.split()[0] for line in open(dictionary_file)]
        self.labels = [line.split()[1] for line in open(dictionary_file)]

    def __getitem__(self, index):
        image = prepare_image(self.image_files[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

    def read(self, batch_size, shuffle=False):
        """Read (image, label) pairs in batch"""
        order = list(range(len(self)))
        if shuffle:
            random.shuffle(order)
        for batch in make_batch(order, batch_size):
            images, labels = [], []
            for index in batch:
                image, label = self[index]
                images.append(image)
                labels.append(label)
            yield np.stack(images).astype(np.float32), np.stack(labels).astype(np.int32).reshape((batch_size,))


def update_dims(height, width, kernel, stride, pad):
    new_height = ((height - kernel + 2*pad)//stride) + 1
    new_width = ((width - kernel + 2*pad)//stride) + 1
    return new_height, new_width


# define model architecture in terms of layers
def AddLeNetModel(model, data):
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
    optimizer.build_sgd(
        model,
        base_learning_rate=0.1,
        policy="step",
        stepsize=1,
        gamma=0.999,
    )


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
    name="train_net", arg_scope=arg_scope)
softmax = AddLeNetModel(train_model, "data")
AddTrainingOperators(train_model, softmax, "label")
#AddBookkeepingOperators(train_model)


# Validation model
val_model = model_helper.ModelHelper(
    name="val_net", arg_scope=arg_scope, init_params=False)
softmax = AddLeNetModel(val_model, "data")
AddAccuracy(val_model, softmax, "label")


# Testing model
test_model = model_helper.ModelHelper(
    name="test_net", arg_scope=arg_scope, init_params=False)
softmax = AddLeNetModel(test_model, "data")
AddAccuracy(test_model, softmax, "label")


# Deployment model
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
# Initialize
train_dataset = MSTAR_Dataset(TRAIN_DICTIONARY)
validation_dataset = MSTAR_Dataset(VALIDATION_DICTIONARY)
test_dataset = MSTAR_Dataset(TEST_DICTIONARY)

for image, label in train_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

for image, label in validation_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

for image, label in test_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# The parameter initialization network only needs to be run once.
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net, overwrite=True)
# initialize and create validation network
workspace.RunNetOnce(val_model.param_init_net)
workspace.CreateNet(val_model.net, overwrite=True)
# variables to track the accuracy & loss
accuracy = np.zeros(training_iters)
loss = np.zeros(training_iters)
# Now, we will manually run the network
for i in range(training_iters):
    for image, label in train_dataset.read(batch_size=training_net_batch_size, shuffle=True):
        workspace.FeedBlob("data", image)
        workspace.FeedBlob("label", label)
        #break
    workspace.RunNet(train_model.net)
    accuracy[i] = workspace.FetchBlob('accuracy')
    loss[i] = workspace.FetchBlob('loss')
    if (i % validation_interval == 0):
        print("Training iter: ", i)
        #run validation
        for image, label in validation_dataset.read(batch_size=validation_images, shuffle=True):
            workspace.FeedBlob("data", image)
            workspace.FeedBlob("label", label)
            #break
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
data = workspace.FetchBlob("data")
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
for image, label in test_dataset.read(batch_size=testing_images, shuffle=True):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    #break
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
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())
