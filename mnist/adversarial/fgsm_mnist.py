# MatthewInkawhich
from __future__ import print_function
from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
import lmdb
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core,model_helper,net_drawer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags


def sort_softmax(softmax):
    sorted_softmax = []
    for i, conf in enumerate(softmax[0]):
        sorted_softmax.append((i, conf))
    sorted_softmax.sort(key=lambda tup: tup[1], reverse=True)
    return sorted_softmax

def print_softmax(sorted_softmax):
    print("Prediction:")
    for i, conf in sorted_softmax:
        print('class:', i, '\t', 'confidence:', conf)




#########################################################################
### Handle inputs and configs
#########################################################################
epsilon = .30
TEST_LMDB = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'custom_mnist', 'test_lmdb')
INIT_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mnist_init_net.pb')
PREDICT_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mnist_predict_net.pb')


# Make sure the specified inputs exist
if ((not os.path.exists(TEST_LMDB)) or (not os.path.exists(INIT_NET)) or (not os.path.exists(PREDICT_NET))):
	print ("ERROR: An input was not found")
	exit()

lmdb_env = lmdb.open(TEST_LMDB, readonly=True)
num_images = lmdb_env.stat()['entries']
print('num_images:',lmdb_env.stat()['entries'])




#########################################################################
### Bring up network
#########################################################################
##### Create a model object (using model helper)
arg_scope = {"order": "NCHW"}
test_model = model_helper.ModelHelper(name="test_model", arg_scope=arg_scope, init_params=False)


##### Create the "data" blob that the net looks for as input
data_uint8, label = test_model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=1, db=TEST_LMDB, db_type='lmdb')
#data, label = test_model.TensorProtosDBInput([], ["data", "label"], batch_size=1, db=TEST_LMDB, db_type='lmdb')
data = test_model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
data = test_model.Scale(data,data,scale=float(1./256))
data = test_model.StopGradient(data,data)


##### Add the pretrained model stuff (info from the pbs) to the model helper object
# Populate the model obj with the init net stuff, which provides the parameters for the model
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())
tmp_param_net = core.Net(init_net_proto)
test_model.param_init_net = test_model.param_init_net.AppendNet(tmp_param_net)

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
test_model.net = test_model.net.AppendNet(tmp_predict_net)

predictor = workspace.Predictor(init_net_proto, predict_net_proto)


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




#########################################################################
### Run initial inference
#########################################################################
# Run a test pass on the test net (same as in MNIST tutorial)
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)
# initialize total and success for success rate
success = 0
total = 0
for _ in range(num_images):
    workspace.RunNet(test_model.net.Proto().name)
    data = workspace.FetchBlob('data')
    softmax = workspace.FetchBlob('softmax')
    label = workspace.FetchBlob('label')

    #print("softmax.shape", softmax.shape)
    sorted_softmax = sort_softmax(softmax)
    print('\nOriginal image results:')
    print_softmax(sorted_softmax)

    ##### If initial prediction is incorrect, no need to attack
    if (sorted_softmax[0][0] != label[0]):
        print("INCORRECT INITIAL PREDICTION; SKIPPING IMAGE")
        continue

    # if initial prediction is correct, increment total
    total += 1




    #########################################################################
    ### Apply FGSM
    #########################################################################
    I = data[0][0]
    #print('\ndata:', I.shape)
    #print('\nsoftmax: ', softmax[0])
    #print('\nlabel: ', label[0])
    data_grad = workspace.FetchBlob('data_grad')
    print("data_grad.shape", data_grad.shape)
    #print("data_grad", data_grad)

    h, w = I.shape
    #print('h:', h, 'w:', w)

    # Create noise matrix
    epsilon_sign_matrix = np.zeros(shape=I.shape)
    for i in range(w):
        for j in range(h):
            if (data_grad[0,0,i,j] < 0):
                epsilon_sign_matrix[i,j] = -epsilon
            else:
                epsilon_sign_matrix[i,j] = epsilon

    # Add noise matrix to original image
    fgs_img = I + epsilon_sign_matrix

    # Check bounds
    for i in range(w):
        for j in range(h):
            if (fgs_img[i,j] < 0):
                fgs_img[i,j] = 0
            elif (fgs_img[i,j] > 1):
                fgs_img[i,j] = 1


    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(I, cmap='gray')
    # axarr[1].imshow(fgs_img, cmap='gray')
    # plt.show()
    plt.imshow(fgs_img, cmap='gray')
    plt.axis('off')
    #plt.savefig("~/DukeML/junk/im" + str(total) + ".png")
    plt.show()
    plt.imshow(epsilon_sign_matrix, cmap='gray')
    plt.show()
    continue


    # Classify FGSM image
    # Add an axis in the channel dimension
    img = fgs_img[np.newaxis, :, :].astype(np.float32)
    img = img[np.newaxis, :, :, :].astype(np.float32)
    # run the net and return prediction
    results = predictor.run([img])
    results = np.asarray(results)
    results = results[0,:,:]
    results = sort_softmax(results)
    print('\nFGSM results:')
    print_softmax(results)
    print('\n\n\n')

    if results[0][0] != label[0]:
        success += 1

print('\n\nSuccess rate: ' + str(success) + '/' + str(total) + '\t' '{0:.2f}'.format(success/total))
