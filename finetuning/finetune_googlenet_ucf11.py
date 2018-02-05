# NAI
# This script shows how to finetune GoogLeNet (downloaded from modelzoo) on jpgs
#   created from UCF11 dataset. Here, finetuning means using GoogLeNet as a feature
#   extractor and retraining/reshaping the final layer to output 11 classes rather
#   than 1000.
#   In this case, the training and testing data are in lmdbs already, resized to 227x227
# Adapted from: https://nbviewer.jupyter.org/gist/kyamagu/6cff70840c10ca374e069a3a7eb00cb4

import os
import numpy as np
from caffe2.python import core, workspace, model_helper, optimizer, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt
from caffe2.python.predictor import mobile_exporter


# Important: from looking at the protobuf for GoogLeNet's predict_net.pb
#   external input: 'data'
#   external output: 'prob'
#   Name of last layer: 'loss3/classifier_w' and 'loss3/classifier_b'

##################################################################################
# Inputs
TRAIN_LMDB = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11-lmdb/UCF11-test-lmdb")
PREDICT_NET = os.path.join(os.path.expanduser('~'),"DukeML/models/bvlc_googlenet/predict_net.pb")
INIT_NET = os.path.join(os.path.expanduser('~'),"DukeML/models/bvlc_googlenet/init_net.pb")
train_iters = 50

##################################################################################
# Create a new model helper object that we can load the nets into
arg_scope = {"order": "NCHW"}
my_model = model_helper.ModelHelper(name="squeezenet_for_ucf11", arg_scope=arg_scope, init_params=False)

##################################################################################
# Add the data layer for the train lmdb
data_uint8, label = my_model.TensorProtosDBInput([], ['data_uint8', 'label'], batch_size=10, db=TRAIN_LMDB, db_type='lmdb')
data = my_model.Cast(data_uint8, 'data', to=core.DataType.FLOAT)
data = my_model.Scale(data, data, scale=float(1./256.))
# enforce a stopgradient because we do not need the gradient of the data for the backward pass
data = my_model.StopGradient(data,data)

##################################################################################
# Load the predict net
# Note this is the same as importing any pb file. There are no structural changes required
#	in this file so we can immediately add it as the '.net' member of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
my_model.net = my_model.net.AppendNet(tmp_predict_net)
#my_model.Squeeze('prob', 'softmax', dims=[2,3]) # (?)

##################################################################################
# Load the init net
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())

# Define the parameters to learn in the model
# Since we are no longer using all 1000 classes of imagenet, we have to reset these two layers
# 	to have the output dimensions = # of classes in UCF11 = 11 classes
params_to_learn = ['loss3/classifier_w', 'loss3/classifier_b']
# Iterate through all of the ops in the init_net
for op in init_net_proto.op:
    param_name = op.output[0]
    # If the current op is the parameter we want to learn
    if param_name in params_to_learn:
        print "here"
    	# Set tags to WEIGHT or BIAS depending on what conv10_w or conv10_b
        tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
        # (?) - why is the shape the same as op.arg[0], shouldnt we change the dimension here?
        my_model.create_param(param_name=param_name, shape=op.arg[0], initializer=initializers.ExternalInitializer(), tags=tags)

# Print the params to learn in the ops of the init net. When you find one that is in
#   params_to_learn, mark the index and we will delete it later.
indx_to_remove = []
for i in range(len(init_net_proto.op)):
    print "\n******************************"
    print "OP: ", i
    print "******************************"
    print "OP_NAME: ",init_net_proto.op[i].name
    print "OP_INPUT: ",init_net_proto.op[i].input 
    print "OP_OUTPUT: ",init_net_proto.op[i].output
    if init_net_proto.op[i].output[0] in params_to_learn:
        indx_to_remove.append(i)
    print "OP_SHAPE: ",init_net_proto.op[i].arg[0]


print "Index to remove:",indx_to_remove

#exit()

#print indx_to_remove
#for i in indx_to_remove:
#    init_net_proto.op.pop(i)
init_net_proto.op.pop(115)
init_net_proto.op.pop(114)

# create a temporary net for the init net
out_dim = 11 # new number of classes
tmp_init_net = core.Net(init_net_proto)

# Finally, add this modified init net to the model helper as the param init net
#	Recall, the reason we append here is because we already added the data stuff to the model
my_model.param_init_net = my_model.param_init_net.AppendNet(tmp_init_net)

# Now, fill in the empty conv10_w and conv10_b data that we specifically said
#	we would initialize "Externally"
# Note, the shape information is extracted from the op print out above.
#	We maintain the shape in the other dimensions, but change the 
#	shape in the first dimension from 1000 to 11
my_model.param_init_net.XavierFill([],'loss3/classifier_w',shape=[out_dim, 1024])
my_model.param_init_net.ConstantFill([], 'loss3/classifier_b', shape=[out_dim])

#i = my_model.StopGradient('inception_5b/pool_proj_w','inception_5b/pool_proj_w')

# Print the new init net to see that the shape has changed
for i in range(len(my_model.param_init_net.Proto().op)):
    print "\n******************************"
    print "OP: ", i
    print "******************************"
    print "OP_NAME: ",my_model.param_init_net.Proto().op[i].name
    print "OP_INPUT: ",my_model.param_init_net.Proto().op[i].input 
    print "OP_OUTPUT: ",my_model.param_init_net.Proto().op[i].output
    print "OP_SHAPE: ",my_model.param_init_net.Proto().op[i].arg[0]

for param in my_model.params:
    print param

tmp = workspace.InferShapesAndTypes([my_model.param_init_net])
for t in tmp[0]:
    print t

#exit()

workspace.ResetWorkspace()

##################################################################################
# Add the training operators
xent = my_model.LabelCrossEntropy(['prob', 'label'], 'xent')
loss = my_model.AveragedLoss('xent', 'loss')
brew.accuracy(my_model, ['prob', 'label'], 'accuracy')
my_model.AddGradientOperators(['loss'])
opt = optimizer.build_sgd(my_model, base_learning_rate=0.1)
for param in my_model.GetOptimizationParamInfo():
    opt(my_model.net, my_model.param_init_net, param)

exit()

##################################################################################
# Run the training
workspace.RunNetOnce(my_model.param_init_net)
workspace.CreateNet(my_model.net, overwrite=True)

total_iters = train_iters
accuracy = np.zeros(total_iters)
loss = np.zeros(total_iters)

for i in range(total_iters):
	workspace.RunNet(my_model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	print "accuracy: ", accuracy[i]
	print "loss: ", loss[i]

plt.plot(loss, 'b', label="loss")
plt.plot(accuracy, 'r', label="accuracy")
plt.legend(loc="upper right")
plt.show()

exit()

##################################################################################
# Save the newly finetuned model
deploy_model = model_helper.ModelHelper("finetuned_squeezenet_ucf11_deploy", arg_scope=arg_scope, init_params=False)
deploy_model.net = core.Net(predict_net_proto)
my_model.Squeeze('softmaxout', 'softmax', dims=[2,3])

workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True)

init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())
