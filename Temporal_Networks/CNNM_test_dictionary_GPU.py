##################################################################################
# NAI
#
# This script is what we will use for testing our trained model based on a labeled
#   test dictionary. This does not format a csv for submission but can be used to
#   quickly see how the model we just trained does on our test set. Note, the test
#   dictionary is labeled, so this will output a % ACCURACY
#
# This script supports GPU testing because it does not use the predictor, rather
#   it brings up a net and manually feeds blobs into the workspace.
#
##################################################################################

# import dependencies
print "Import Dependencies..."
import matplotlib.pyplot as plt
import numpy as np 
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import random
import skimage.io
from skimage.color import rgb2gray
import JesterDatasetHandler as jdh

##################################################################################
# Gather Inputs
test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/SmallTestDictionary_5class.txt")
PREDICT_NET = "CNNM_jester_predict_net.pb"
INIT_NET = "CNNM_10epoch_jester_init_net.pb"

gpu_no = 0
device_opts = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CUDA)


##################################################################################
# Bring in trained model

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW", "gpu_id": gpu_no, "use_cudnn": True}
#arg_scope = {"order":"NCHW"}
test_model = model_helper.ModelHelper(name="CNNM_jester_test", arg_scope=arg_scope)

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
test_model.net = tmp_predict_net

# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
test_model.param_init_net = tmp_param_net


test_model.param_init_net.RunAllOnGPU()
test_model.net.RunAllOnGPU()
##################################################################################
# Loop through the test dictionary and run the inferences

# Confusion Matrix
cmat = np.zeros((5,5))

# Initialization.
test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)

# Prime the workspace with some data and run init net once
for image, label in test_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)

num_correct = 0
total = 0

# Cycle through the test dictionary once, with batch size = 1, meaning we only consider one stack at a time
for stack, label in test_dataset.read(batch_size=1):

    # Run the stack through the predictor and get the result array
    workspace.FeedBlob("data", stack, device_option=device_opts)
    workspace.FeedBlob("label", label, device_option=device_opts)
    workspace.RunNet(test_model.net)
    results = workspace.FetchBlob('softmax')[0]

    print results
    # Get the top-1 prediction
    max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))

    print "Prediction: ", max_index
    print "Confidence: ", max_value
    
    # Update confusion matrix
    cmat[label,max_index] += 1

    if max_index == label:
        num_correct += 1

    total += 1
    
print "\n**************************************"
print "Total Tests = {}".format(total)
print "# Correct = {}".format(num_correct)
print "Accuracy = {}".format(num_correct/float(total))

'''
# Plot confusion matrix
fig = plt.figure()
#plt.clf()
plt.tight_layout()
ax = fig.add_subplot(111)
#ax.set_aspect(1)
res = ax.imshow(cmat, cmap=plt.cm.rainbow,interpolation='nearest')

width, height = cmat.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cmat[x,y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')

#cb = fig.colorbar(res)
classes = ['Swipe Left', 'Swipe Right', 'Swipe Down', 'Swipe Up', 'Push Away']
plt.xticks(range(width), classes, rotation=17)
plt.yticks(range(height), classes, rotation=17)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
plt.title('Jester 5-class Confusion Matrix')
plt.show()
'''

