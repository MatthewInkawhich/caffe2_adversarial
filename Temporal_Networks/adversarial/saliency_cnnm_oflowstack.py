##################################################################################
# NAI
#
# This file contains the FGSM implementation for the CNNM network trained on
#   optical flow stacks. Rather than computing the FGSM stacks and saving them
#   to disk, this script takes in a dictionary of videos, creates all possible
#   optical flow stacks from the videos, then for each one, run the stack through
#   the network, gets the data_grad blob from the workspace, applies the FGSM
#   given the datagrad, then retests the perturbed image to check the success
#   of the attack.
#
##################################################################################

from __future__ import print_function
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import lmdb
import operator
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
import JesterDatasetHandler as jdh

##################################################################################
### Handle inputs and configs
##################################################################################
N = 100
TEST_DICT = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'VerySmallTestDictionary_5class.txt')
INIT_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'Temporal_Networks', 'CNNM_2epoch_jester_init_net.pb')
PREDICT_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'Temporal_Networks', 'CNNM_jester_predict_net.pb')


# Make sure the specified inputs exist
if ((not os.path.exists(TEST_DICT)) or (not os.path.exists(INIT_NET)) or (not os.path.exists(PREDICT_NET))):
	print ("ERROR: An input was not found")
	exit()

##################################################################################
### Bring up network
##################################################################################
##### Create a model object (using model helper)
arg_scope = {"order": "NCHW"}
test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)

##### Add the pretrained model stuff (info from the pbs) to the model helper object
# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
#test_model.param_init_net = test_model.param_init_net.AppendNet(tmp_param_net)
test_model.param_init_net = tmp_param_net

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
#test_model.net = test_model.net.AppendNet(tmp_predict_net)
test_model.net = tmp_predict_net

##### Externally initialize params so we can extract gradients
for i,op in enumerate(init_net_proto.op):
	param_name = op.output[0]
	if param_name != 'data':
		assert(op.arg[0].name == "shape")
		tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
		test_model.create_param(param_name=op.output[0], shape=op.arg[0].ints, initializer=initializers.ExternalInitializer(), tags=tags)


##### Add the "training operators" to the model
xent = test_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = test_model.AveragedLoss(xent, 'loss')
test_model.AddGradientOperators([loss])

##################################################################################
### Run
##################################################################################

# Initialize Dataset Object
test_dataset = jdh.Jester_Dataset(dictionary_file=TEST_DICT,seq_size=10)

# Prime the workspace with some data so we can run init net once
for image, label in test_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# Run a test pass on the test net (same as in MNIST tutorial)
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)

num_correct = 0
num_succeed = 0
num_attacks = 0
total = 0

# Cycle through the test dictionary once, with batch size = 1, meaning we only consider one stack at a time
#   and each stack is considered only once
for stack, label in test_dataset.read(batch_size=1):

    # Keep track of how many stacks we have looked at
    total += 1

    # Run the stack through the network which will produce a softmax vector and populate the gradient blobs
    # stack.shape = [1, 20, 100, 100]
    workspace.FeedBlob("data", stack)
    workspace.FeedBlob("label", label)
    workspace.RunNet(test_model.net)
    
    # Fetch relevant blobs from the workspace
    curr_loss = workspace.FetchBlob('loss')
    softmax = workspace.FetchBlob('softmax')
    lab = workspace.FetchBlob('label')

    # Get the top-1 prediction
    max_index, max_value = max(enumerate(softmax[0]), key=operator.itemgetter(1))

    print ("Curr_loss = ",curr_loss)
    print ("Softmax = ",softmax)
    print ("label = ",label)
    print ("workspace lab = ",lab)
    print ("max ind = ",max_index)
    
    ##### If initial prediction is incorrect, no need to attack
    if (max_index != label[0]):
        print("INCORRECT INITIAL PREDICTION; SKIPPING IMAGE")
        continue
    
    # if initial prediction is correct, increment num_attacks
    num_attacks += 1

    ##############################################################
    # Saliency Map Attack
    ##############################################################
    # Create a copy of the stack to perturb
    pstack = np.copy(stack)
    binary_noise_field = np.zeros((100,100))
    
    # Get the data grad blob from the workspace, this is the same dimensionality as the input stack
    #   and is the gradient of the loss with respect to the data
    data_grad = workspace.FetchBlob('data_grad')
    print("data_grad.shape", data_grad.shape)
    print("image.shape", stack.shape)

    # Take the element-wise absolute value of the gradient of loss w.r.t data to
    #     approximate the saliency map [https://arxiv.org/pdf/1312.6034.pdf]
    saliency_map = np.absolute(data_grad)

    # Find the coordinates of the N largest (i.e. most influential/salient features)
    indices =  np.argpartition(saliency_map.flatten(), -N)[-N:]
    locs =  np.vstack(np.unravel_index(indices, saliency_map.shape)).T

    # Perturb the original optical flow stack at those locations
    for l in locs:
        ## Specific perturbation method applied here!!
        # Perturb Method 1: Negate the optical flow stack at each location
        pstack[l[0], l[1], l[2], l[3]] *= -1
        binary_noise_field[l[2],l[3]] = 1

    # Clip the perturbed stack to keep the same distribution
    pstack[pstack < 0] = 0
    pstack[pstack > 1] = 1

    ##############################################################
    ##############################################################

    # TEST THE ADV EXAMPLE TO SEE IF IT WORKED
    workspace.FeedBlob("data", pstack)
    workspace.FeedBlob("label", label)
    workspace.RunNet(test_model.net)

    softmax = workspace.FetchBlob('softmax')
    lab = workspace.FetchBlob('label')

    # Get the top-1 prediction
    max_index, max_value = max(enumerate(softmax[0]), key=operator.itemgetter(1))


    if (max_index != label[0]):
        print("SUCCESSFUL ADV IMAGE")
        num_succeed += 1
    else:
        print ("FAILED TO CREATE ADVERSARIAL EXAMPLE")
        num_correct += 1



    plt.subplot(1,5,1)
    plt.title("Orig oflow")
    plt.imshow(stack[0,0,:,:],cmap='gray')
    plt.subplot(1,5,2)
    plt.title("Data Grad")
    plt.imshow(data_grad[0,0,:,:],cmap='gray')
    plt.subplot(1,5,3)
    plt.title("Saliency Map")
    plt.imshow(saliency_map[0,0,:,:],cmap='gray')
    plt.subplot(1,5,4)
    plt.title("Noise Field")
    plt.imshow(binary_noise_field,cmap='gray')
    plt.subplot(1,5,5)
    plt.title("perturbed oflow")
    plt.imshow(pstack[0,0,:,:],cmap='gray')
    plt.show()

    exit()


print ("\n**************************************")
print ("N = ",N)
print ("Total Stacks Tested = ",total)
print ("Number of attacks attempted = ",num_attacks)
print ("Number of successful attacks = ",num_succeed)
print ("Old Accuracy = ",num_attacks/float(total))
print ("Adv Accuracy = ",num_correct/float(total))


