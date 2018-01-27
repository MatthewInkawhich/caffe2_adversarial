# NAI
# This script shows how to finetune squeezenet (downloaded from modelzoo) on jpgs
#   created from UCF11 dataset. Here, finetuning means using squeezenet as a feature
#   extractor and retraining/reshaping the final layer to output 11 classes rather
#   than 1000.
#   In this case, the training and testing data are in lmdbs already, resized to 227x227
# Adapted from: https://nbviewer.jupyter.org/gist/kyamagu/6cff70840c10ca374e069a3a7eb00cb4
import os, glob, random
import numpy as np
from caffe2.python import core, workspace, model_helper, optimizer, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt
from caffe2.python.predictor import mobile_exporter
import skimage.io
import skimage.transform

# Need to modify Squeezenet so we can use the pretrained model with our dataset
# Need to change how to initialize the model parameters and the output shape
# We will do this using the model_helper
# Basic Steps:
#   - load predict net
#   - load init net
#   - add training operators

# Important: from looking at the protobuf for squeezenet's predict_net.pb
#   external input: data
#   external output: softmaxout
#   Name of last layer: 'conv10_w' and 'conv10_b'

##################################################################################
# Inputs
#TRAIN_LMDB = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11-lmdb/UCF11-test-lmdb")
TRAIN_DICTIONARY = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/TestDictionary_UCF11_updated_jpg_5FPS.txt")
KEY_FILE = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/KEYFILE_UCF11.txt")
PREDICT_NET = os.path.join(os.path.expanduser('~'),"DukeML/models/squeezenet/predict_net.pb")
INIT_NET = os.path.join(os.path.expanduser('~'),"DukeML/models/squeezenet/init_net.pb")
train_iters = 50










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
    img = rescale(img, 227, 227)
    img = crop_center(img, 227, 227)
    img = img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
    img = img[(2, 1, 0), :, :]                 # RGB to BGR color order
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

    print key_dict
    return key_dict


class UCF11_Dataset(object):
    def __init__(self, dictionary_file=TRAIN_DICTIONARY):
        self.categories = extract_categories(KEY_FILE)
        self.image_files = [line.split()[0] for line in open(dictionary_file)]
        self.labels = [line.split()[1] for line in open(dictionary_file)]
        
    def __getitem__(self, index):
        image = prepare_image(self.image_files[index])
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.labels)
    
    def read(self, batch_size=50, shuffle=True):
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



'''
u = UCF11_Dataset(TRAIN_DICTIONARY)

print u.categories
print u.image_files
print len(u.image_files)
print len(u.labels)

exit()
'''












##################################################################################
# Create a new model helper object that we can load the nets into
arg_scope = {"order": "NCHW"}
my_model = model_helper.ModelHelper(name="squeezenet_for_ucf11", arg_scope=arg_scope, init_params=False)

'''
##################################################################################
# Add the data layer for the train lmdb
data_uint8, label = my_model.TensorProtosDBInput([], ['data_uint8', 'label'], batch_size=10, db=TRAIN_LMDB, db_type='lmdb')
data = my_model.Cast(data_uint8, 'data', to=core.DataType.FLOAT)
data = my_model.Scale(data, data, scale=float(1./256.))
# enforce a stopgradient because we do not need the gradient of the data for the backward pass
data = my_model.StopGradient(data,data)
'''

##################################################################################
# Load the predict net
# Note this is the same as importing any pb file. There are no structural changes required
#	in this file so we can immediately add it as the '.net' member of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
#my_model.net = my_model.net.AppendNet(tmp_predict_net)
my_model.net = tmp_predict_net
my_model.Squeeze('softmaxout', 'softmax', dims=[2,3])

##################################################################################
# Load the init net
init_net_proto = caffe2_pb2.NetDef()
with open(INIT_NET, "rb") as f:
    init_net_proto.ParseFromString(f.read())

# Define the parameters to learn in the model
# Since we are no longer using all 1000 classes of imagenet, we have to reset these two layers
# 	to have the output dimensions = # of classes in UCF11 = 11 classes
# We do this in layer conv10 because this is the last layer where we can control the dimensionality.
#	In the model, after conv10 there is a max pool followed by the softmax.
params_to_learn = ['conv10_w', 'conv10_b']
# Iterate through all of the ops in the init_net
for op in init_net_proto.op:
    param_name = op.output[0]
    # If the current op is the parameter we want to learn [i.e. 'conv10_w', 'conv10_b']
    if param_name in params_to_learn:
    	# Set tags to WEIGHT or BIAS depending on what conv10_w or conv10_b
        tags = (ParameterTags.WEIGHT if param_name.endswith("_w") else ParameterTags.BIAS)
        # (?) - what is this doing? - what is a parameter in a model?
        # since we are going to delete these two, we must now create new versions of these
        # 	and specify that they will be initialized externally
        # (?) - why is the shape the same as op.arg[0], shouldnt we change the dimension here?
        my_model.create_param(param_name=param_name, shape=op.arg[0], initializer=initializers.ExternalInitializer(), tags=tags)

# Remove conv10_w, conv10_b initializers at (50,51)
# To get the numbers, 50 and 51, must look at the ops in the init_net model.
#	If we print the ops from the pb using the following:
# 		$ python print_pb_verbose.py ../../models/squeezenet/init_net.pb
# 	We see that op[50] outputs 'conv10_w' and op[51] outputs 'conv10_b', thus, these
#	are the ops we desire to remove and update

# Print the info for the conv10 ops we are about to remove. This verifies that
#	ops 50 and 51 in the init_net are in-fact conv10_w and conv10_b and it also
#	shows us the shape information, which we must change from 1000 to 11
print "The ops we are about to remove and replace..."
for i in [50,51]:
	print "\n******************************"
	print "OP: ", i
	print "******************************"
	print "OP_NAME: ",init_net_proto.op[i].name
	print "OP_INPUT: ",init_net_proto.op[i].input 
	print "OP_OUTPUT: ",init_net_proto.op[i].output
	print "OP_SHAPE: ",init_net_proto.op[i].arg[0]

# Actually remove the old ops for conv10_w and conv10_b
init_net_proto.op.pop(51)
init_net_proto.op.pop(50)

# create a temporary net for the init net
out_dim = 11 # new number of classes
tmp_init_net = core.Net(init_net_proto)

# Finally, add this modified init net to the model helper as the param init net
#	Recall, the reason we append here is because we already added the data stuff to the model
#my_model.param_init_net = my_model.param_init_net.AppendNet(tmp_init_net)
my_model.param_init_net = tmp_init_net

# Now, fill in the empty conv10_w and conv10_b data that we specifically said
#	we would initialize "Externally"
# Note, the shape information is extracted from the op print out above.
#	We maintain the shape in the other dimensions, but change the 
#	shape in the first dimension from 1000 to 11
my_model.param_init_net.XavierFill([],'conv10_w',shape=[out_dim, 512, 1, 1])
my_model.param_init_net.ConstantFill([], 'conv10_b', shape=[out_dim])

##################################################################################
# Add the training operators
xent = my_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = my_model.AveragedLoss('xent', 'loss')
brew.accuracy(my_model, ['softmax', 'label'], 'accuracy')
my_model.AddGradientOperators([loss])
opt = optimizer.build_sgd(my_model, base_learning_rate=0.1)
for param in my_model.GetOptimizationParamInfo():
    opt(my_model.net, my_model.param_init_net, param)


print "\n***************** PRINTING MODEL PARAMS *******************"
for param in my_model.params:
    print "PARAM: ",param

# Params that will be optimized during training
print "\n***************** OPT PARAM INFO *******************"
print my_model.GetOptimizationParamInfo()

#exit()

##################################################################################
# Run the training

# Initialization.
train_dataset = UCF11_Dataset(TRAIN_DICTIONARY)
for image, label in train_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break
workspace.RunNetOnce(my_model.param_init_net)
workspace.CreateNet(my_model.net, overwrite=True)

# Main loop.
batch_size = 20
print_freq = 1
losses = []
for epoch in range(5):
    for index, (image, label) in enumerate(train_dataset.read(batch_size)):
        workspace.FeedBlob("data", image)
        workspace.FeedBlob("label", label)
        workspace.RunNet(my_model.net)
        accuracy = float(workspace.FetchBlob("accuracy"))
        loss = workspace.FetchBlob("loss").mean()
        losses.append(loss)
        if index % print_freq == 0:
            print("[{}][{}/{}] loss={}, accuracy={}".format(epoch, index, int(len(train_dataset) / batch_size),loss, accuracy))







exit()

'''
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

'''

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
