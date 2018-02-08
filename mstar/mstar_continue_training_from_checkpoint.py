# MatthewInkawhich
# This script is used to continue training the MSTAR net from a LMDB checkpoint

from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter



########################################################################
# CONFIGS
########################################################################
training_lmdb = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'mstar64', 'train_lmdb')
saved_checkpoint = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_output', 'mstar_lenet_checkpoint_00200.lmdb')
predict_net = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'mstar_output', 'mstar_predict_net.pb')
training_iters = 100

# Make sure the training lmdb exists
if not os.path.exists(training_lmdb):
	print "ERROR: train lmdb NOT found"
	exit()



########################################################################
# CREATE AND LOAD CHECKPOINT DATA INTO MODEL
########################################################################
arg_scope = {"order":"NCHW"}

# Create new model
model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope, init_params=False)

# Add data layer to model
data_uint8, label = model.TensorProtosDBInput([], ["data_uint8", "label"], batch_size=100, db=training_lmdb, db_type='lmdb')
data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
data = model.Scale(data,data,scale=float(1./256))
data = model.StopGradient(data,data)

# Populate the model obj with the predict net def
predict_net_proto = caffe2_pb2.NetDef()
with open(predict_net, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
model.net = model.net.AppendNet(tmp_predict_net)

# Load params and blobs from checkpoint lmdb
workspace.RunOperatorOnce(
      core.CreateOperator("Load", [], [], absolute_path=1, db=saved_checkpoint, db_type="lmdb", load_all=1))

# Add the "training operators" to the model
xent = model.LabelCrossEntropy(['softmax', 'label'], 'xent')
# compute the expected loss
loss = model.AveragedLoss(xent, "loss")
# track the accuracy of the model
accuracy = brew.accuracy(model, ['softmax', 'label'], "accuracy")
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
# CONTINUE TRAINING
########################################################################
workspace.RunNetOnce(model.param_init_net)
workspace.CreateNet(model.net, overwrite=True)

accuracy = np.zeros(training_iters)
loss = np.zeros(training_iters)

for i in range(training_iters):
	workspace.RunNet(model.net)
	accuracy[i] = workspace.FetchBlob('accuracy')
	loss[i] = workspace.FetchBlob('loss')
	if i % 20 == 0:
		print "i:", i
		print "accuracy: ", accuracy[i]
		print "loss: ", loss[i]



########################################################################
# SHOW TEST RESULTS
########################################################################
plt.plot(loss, 'b', label="loss")
plt.plot(accuracy, 'r', label="accuracy")
plt.legend(loc="upper right")
plt.xlabel("Iteration")
plt.ylabel("Loss and Accuracy")
plt.title("Loss and Accuracy through training")
plt.show()
