##################################################################################
# NAI
#
# This script is what we will use for testing our trained model based on a labeled
#   test dictionary. This does not format a csv for submission but can be used to
#   quickly see how the model we just trained does on our test set. Note, the test
#   dictionary is labeled, so this will output a % ACCURACY
#
##################################################################################

# import dependencies
print "Import Dependencies..."
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import operator
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
import random
import skimage.io
from skimage.color import rgb2gray
import JesterDatasetHandler as jdh
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'lib'))
import model_defs

##################################################################################
# Gather Inputs
test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TestDictionary_5class.txt")
# predict_net = "CNNM_jester_predict_net.pb"
# init_net = "CNNM_4epoch_jester_init_net.pb"

init_net = os.path.join(os.path.expanduser('~'),"DukeML", "models", "CNNM_3", "CNNM_3epoch_jester_init_net.pb")
predict_net = os.path.join(os.path.expanduser('~'),"DukeML", "models", "CNNM_3", "CNNM_jester_predict_net.pb")
saved_checkpoint = os.path.join(os.path.expanduser('~'), "DukeML", "models", "CNNM_3", "cnnm_checkpoint_16000.lmdb")
device_opts=core.DeviceOption(caffe2_pb2.CPU, 0)
image_width = 100
image_height = 100
image_channels = 20
num_classes = 5


########################################################################
# CREATE AND LOAD CHECKPOINT DATA INTO MODEL
########################################################################
arg_scope = {"order":"NCHW"}

# Create new model
model = model_helper.ModelHelper(name="CNNM_test_model", arg_scope=arg_scope, init_params=False)

# # Add data layer to model
# data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=100, db=training_lmdb, db_type='lmdb')
# data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
# data = model.Scale(data,data,scale=float(1./256))
# data = model.StopGradient(data,data)

# init_net_proto = caffe2_pb2.NetDef()
# with open(init_net, "rb") as f:
#     init_net_proto.ParseFromString(f.read())
#     init_net_proto.device_option.CopyFrom(device_opts_cpu)
# tmp_param_net = core.Net(init_net_proto)
# model.param_init_net = model.param_init_net.AppendNet(tmp_param_net)

# Populate the model obj with the predict net def
predict_net_proto = caffe2_pb2.NetDef()
with open(predict_net, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
    predict_net_proto.device_option.CopyFrom(device_opts)
tmp_predict_net = core.Net(predict_net_proto)
model.net = model.net.AppendNet(tmp_predict_net)

#print(model.net.Proto())

# Load params and blobs from checkpoint lmdb
workspace.RunOperatorOnce(
      core.CreateOperator("Load", [], [], absolute_path=1, db=saved_checkpoint, db_type="lmdb", keep_device=1, load_all=1))

##### Externally initialize params so we can extract gradients
# for i,op in enumerate(init_net_proto.op):
# 	param_name = op.output[0]
# 	if param_name != 'data':
# 		print "param_name:", param_name
# 		assert(op.arg[0].name == "shape")
# 		tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
# 		model.create_param(param_name=op.output[0], shape=op.arg[0].ints, initializer=initializers.ExternalInitializer(), tags=tags)


# Add the "training operators" to the model
softmax = model_defs.Add_CNN_M(model,'data', device_opts)
xent = model.LabelCrossEntropy([softmax, 'label'], 'xent')
loss = model.AveragedLoss(xent, "loss")
accuracy = brew.accuracy(model, [softmax, 'label'], "accuracy")
model.AddGradientOperators([loss])

# Instatiate test_dataset object
test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)

# Prime the workspace with some data so we can run init net once
for image, label in test_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# run the param init network once
workspace.RunNetOnce(model.param_init_net)
# create the network
workspace.CreateNet(model.net, overwrite=True)

##################################################################################
# Bring up the network from the .pb files
# with open(init_net) as f:
#     init_net = f.read()
# with open(predict_net) as f:
#     predict_net = f.read()
#
# p = workspace.Predictor(init_net, predict_net)


##################################################################################
# Loop through the test dictionary and run the inferences

# Confusion Matrix
# cmat = np.zeros((5,5))
#
# # Initialization.
# test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)
#
# num_correct = 0
# total = 0
#
# # Cycle through the test dictionary once, with batch size = 1, meaning we only consider one stack at a time
# for stack, label in test_dataset.read(batch_size=1):
#
#     # Run the stack through the predictor and get the result array
#     results = np.asarray(p.run([stack]))[0,0,:]
#     print results
#     # Get the top-1 prediction
#     max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))
#
#     print "Prediction: ", max_index
#     print "Confidence: ", max_value
#
#     # Update confusion matrix
#     cmat[label,max_index] += 1
#
#     if max_index == label:
#         num_correct += 1
#
#     total += 1
#
# print "\n**************************************"
# print "Total Tests = {}".format(total)
# print "# Correct = {}".format(num_correct)
# print "Accuracy = {}".format(num_correct/float(total))
#
# # Plot confusion matrix
# fig = plt.figure()
# #plt.clf()
# plt.tight_layout()
# ax = fig.add_subplot(111)
# #ax.set_aspect(1)
# res = ax.imshow(cmat, cmap=plt.cm.rainbow,interpolation='nearest')
#
# width, height = cmat.shape
#
# for x in xrange(width):
#     for y in xrange(height):
#         ax.annotate(str(cmat[x,y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')
#
# #cb = fig.colorbar(res)
# classes = ['Swipe Left', 'Swipe Right', 'Swipe Down', 'Swipe Up', 'Push Away']
# plt.xticks(range(width), classes, rotation=17)
# plt.yticks(range(height), classes, rotation=17)
# ax.set_xlabel('Predicted Class')
# ax.set_ylabel('True Class')
# plt.title('Jester 5-class Confusion Matrix')
# plt.show()
