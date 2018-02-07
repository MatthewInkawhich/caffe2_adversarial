# NAI

# The architecture here is the CNN-M-2048 that is described for the temporal
#   stream in the original two-stream paper

# import dependencies
print "Import Dependencies..."
from matplotlib import pyplot
import numpy as np 
import os
import glob
import shutil
import random
import caffe2.python.predictor.predictor_exporter as pe 
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import cv2



##################################################################################
# Gather Inputs
train_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/SMALL_TestDictionary_5class.txt")
predict_net_out = "CNNM_jester_predict_net.pb" # Note: these are in PWD
init_net_out = "CNNM_2k_jester_init_net.pb"
training_iters = 2000
checkpoint_iters = 1000




def crop_center(img, new_height, new_width):
    orig_height, orig_width, _ = img.shape
    startx = (orig_width//2) - (new_width//2)
    starty = (orig_height//2) - (new_height//2)
    return img[starty:starty+new_height, startx:startx+new_width]


def resize_image(img, new_height, new_width):
    h, w, _ = img.shape
    if (h < new_height or w < new_width):
        img_data_r = imresize(img, (new_height, new_width))
    else:
        img_data_r = crop_center(img, new_height, new_width)
    return img_data_r

# (?) - MEAN SUBTRACT?
def handle_greyscale(img):
    img = img[:,:,0]
    #img = np.expand_dims(img, axis=0)
    return img

def make_batch(iterable, batch_size=1):
    length = len(iterable)
    for index in range(0, length, batch_size):
        yield iterable[index:min(index + batch_size, length)]

def create_oflow_stack(seq):
	# Given seq of ordered jpgs for the optical flow, read them into a numpy array
	# seq = [ \path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ... ]

	#print "in create oflow stack"
	oflow_stack = np.zeros(shape=(20,100,100))

	# For each of the images in the sequence (which are contiguous optical flow images)
	for i in range(len(seq)):
		
		# Read the image as a color image (BGR) into a numpy array as 32 bit floats
		of_img = cv2.imread(seq[i]).astype(np.float32)
		#print "Shape after reading in : {}".format(of_img.shape)
		
		# Resize the image to 3x100x100
		of_img = resize_image(of_img, 100, 100)
		#print "Shape after Resizing : {}".format(of_img.shape)
		
		of_img = handle_greyscale(of_img)
		#print "Shape after greyscale : {}".format(of_img.shape)
	
		oflow_stack[i,:,:] = of_img

		#print "Printing"
		#print oflow_stack
	return oflow_stack
	#exit()


# Given the input dictionary file, we can make a list of sequences with labels that represent all
#	of the unique optical flow stacks that can be created from the input dictionary
#	The return value is a list of labeled sequences, each subsequence is length 20
def make_list_of_seqs(ifile,seq_size):

	my_list_of_seqs = []

	# Open the input dictionary file for reading
	# Each line contains a path to a directory full of jpgs that correspond
	#   to ONE video and the integer label representing the class for that video
	infile = open(ifile,"rb")

	# ex. line = "/.../jester/20bn-jester-v1/12 5"
	for line in infile:

	    split_line = line.split()

	    # Extract the class from the line
	    label = split_line[1].rstrip()

	    # Extract the path from the line
	    path = split_line[0]

	    # Change the path from the original 20bn-jester-v1 to 20bn-jester-v1-oflow
	    path = path.replace("20bn-jester-v1","20bn-jester-v1-oflow")

	    assert(os.path.exists(path) == True)
	    
	    # Go into each directory and get an array of jpgs in the directory (these are full paths)
	    # Note: we only grab _h here, but we assume the _v exists
	    full_oflow_arr = glob.glob(path + "/*_h.jpg")

	    # Sort the array based on the sequence number [e.x. oflow_00028_00030_13_h.jpg ; seq# = 13]
	    full_oflow_arr.sort(key=lambda x: int(x.split("_")[-2]))

	    # Add the subsequences of length seq_size (usually 10)
	    # for   i < (len(arr) - 10)
	    for i in range(len(full_oflow_arr)-seq_size+1):
	        # Alloc list to store a single sequence of length seq_size
	        single_seq = []
	        # for j < 10
	        for j in range(seq_size):
	            # Append the horizontal version
	            single_seq.append(full_oflow_arr[i+j])
	            # Append the vertical version
	            single_seq.append(full_oflow_arr[i+j].replace("_h.jpg","_v.jpg"))
	        # Add this single sequence to the global list of sequences and the label
	        my_list_of_seqs.append([single_seq, label])

	# randomly shuffle list of contiguous sequences
	random.shuffle(my_list_of_seqs)

	# print total number of sequences
	num_sequences = len(my_list_of_seqs)
	print "Total number of sequences: {}".format(num_sequences)
	print "Finished creating list of sequences!"

	return my_list_of_seqs

class Jester_Dataset(object):
	def __init__(self, dictionary_file=train_dictionary,seq_size=10):
		#self.video_dirs = [line.split()[0] for line in open(dictionary_file)]
		#self.labels = [line.split()[1] for line in open(dictionary_file)]
		self.list_of_seqs = make_list_of_seqs(dictionary_file,seq_size)
   	
   	def __getitem__(self, index):
		single_oflow_seq = self.list_of_seqs[index]
		return single_oflow_seq
    
	def __len__(self):
		return len(self.list_of_seqs)

	def read(self, batch_size=50, shuffle=False):

		"""Read (image, label) pairs in batch"""
		order = list(range(len(self)))

		if shuffle:
			random.shuffle(order)

	    # batch is a list of indexes, with length batch size
	    # i.e. bsize = 4 ; batch = [1,2,3,4] then [5,6,7,8] then [9,10,11,12]
		for batch in make_batch(order, batch_size):

			#print "Current Batch : {}".format(batch)
			oflow_batch, labels = [], []

			for index in batch:

				# Single Seq = [ [\path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ...], 1 ]
				single_seq = self[index]

				# Extract the sequence of .jpgs
				seq = single_seq[0]

				# Extract the label
				label = int(single_seq[1])

				# Create an optical flow stack from the seq images
				# oflow_stack is a np.ndarray with shape (20,100,100)
				oflow_stack = create_oflow_stack(seq)

				oflow_batch.append(oflow_stack)
				labels.append(label)

			yield np.stack(oflow_batch).astype(np.float32), np.stack(labels).astype(np.int32).reshape((batch_size,))







'''
jester = Jester_Dataset(train_dictionary)
for i in range(3):
	print jester.list_of_seqs[i]
	print " "
for image, label in jester.read(batch_size=2, shuffle=False):
	print "hi"

exit()
'''



##################################################################################
# MAIN

print "Entering Main..."


##################################################################################
# Create model helper for use in this script

# specify that input data is stored in NCHW storage order
arg_scope = {"order":"NCHW"}

# create the model object that will be used for the train net
# This model object contains the network definition and the parameter storage
train_model = model_helper.ModelHelper(name="CNNM_jester_train", arg_scope=arg_scope)

##################################################################################
#### Add the model definition the the model object

# CNN-M-2048 [https://arxiv.org/pdf/1405.3531.pdf]
# As mentioned in two stream paper, we omit the norm layer in conv2

# O = ( (W - K + 2P) / S ) + 1

def Add_CNN_M(model,data):

	# Shape here = 20x100x100

	##### CONV-1
	conv1 = brew.conv(model, data, 'conv1', dim_in=20, dim_out=96, kernel=7, stride=2, pad=0)
	#norm1 = brew.lrn(model, conv1, 'norm1', order="NCHW")
	# Shape here = 96x47x47
	pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
	# Shape here = 96x23x23
	relu1 = brew.relu(model, pool1, 'relu1')

	# Shape here = 96x23x23

	##### CONV-2
	conv2 = brew.conv(model, 'relu1', 'conv2', dim_in=96, dim_out=256, kernel=5, stride=2, pad=1)
	# Shape here = 256x11x11
	pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
	# Shape here = 256x5x5
	relu2 = brew.relu(model, pool2, 'relu2')

	# Shape here = 256x5x5

	##### CONV-3
	conv3 = brew.conv(model, 'relu2', 'conv3', dim_in=256, dim_out=512, kernel=3, stride=1, pad=1)
	# Shape here = 512x5x5
	relu3 = brew.relu(model, conv3, 'relu3')

	# Shape here = 512x5x5

	##### CONV-4
	conv4 = brew.conv(model, 'relu3', 'conv4', dim_in=512, dim_out=512, kernel=3, stride=1, pad=1)
	# Shape here = 512x5x5
	relu4 = brew.relu(model, conv4, 'relu4')

	# Shape here = 512x5x5

	##### CONV-5
	conv5 = brew.conv(model, 'relu4', 'conv5', dim_in=512, dim_out=512, kernel=3, stride=1, pad=1)
	# Shape here = 512x5x5
	pool5 = brew.max_pool(model, conv5, 'pool5', kernel=2, stride=2)
	# Shape here = 512x2x2
	relu5 = brew.relu(model, pool5, 'relu5')

	# Shape here = 512x2x2

	fc6 = brew.fc(model, relu5, 'fc6', dim_in=512*2*2, dim_out=4096)
	relu6 = brew.relu(model, fc6, 'relu6')

	# Shape here = 1x4096

	fc7 = brew.fc(model, relu6, 'fc7', dim_in=4096, dim_out=4096)
	relu7 = brew.relu(model, fc7, 'relu7')

	# Shape here = 1x4096

	fc8 = brew.fc(model, relu7, 'fc8', dim_in=4096, dim_out=5)

	# Shape here = 1x5

	softmax = brew.softmax(model,fc8, 'softmax')

	return softmax


# Add the model definition to the model
softmax=Add_CNN_M(train_model, 'data')

##################################################################################
#### Step 3: Add training operators to the model

ITER = brew.iter(train_model, "iter")
train_model.Checkpoint([ITER] + train_model.params, [], db="cnnm_checkpoint_%05d.lmdb", db_type="lmdb", every=checkpoint_iters)


xent = train_model.LabelCrossEntropy(['softmax', 'label'], 'xent')
loss = train_model.AveragedLoss(xent, 'loss')
brew.accuracy(train_model, ['softmax', 'label'], 'accuracy')
train_model.AddGradientOperators([loss])

optimizer.build_sgd(train_model,base_learning_rate=0.1, policy="step", stepsize=1, gamma=0.999)

##################################################################################
#### Run the training procedure

# Initialization.
train_dataset = Jester_Dataset(train_dictionary,10)

# Prime the workspace with some data so we can run init net once
for image, label in train_dataset.read(batch_size=1):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# run the param init network once
workspace.RunNetOnce(train_model.param_init_net)
# create the network
workspace.CreateNet(train_model.net, overwrite=True)


# Set the total number of iterations and track the accuracy and loss
total_iters = training_iters
accuracy = []
loss = []

batch_size = 50
# Manually run the network for the specified amount of iterations
for epoch in range(1):

	for index, (image, label) in enumerate(train_dataset.read(batch_size)):
		workspace.FeedBlob("data", image)
		workspace.FeedBlob("label", label)
		workspace.RunNet(train_model.net) # SEGFAULT HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		curr_acc = workspace.FetchBlob('accuracy')
		curr_loss = workspace.FetchBlob('loss')
		accuracy.append(curr_acc)
		loss.append(curr_loss)
		#print "Iter: {}, loss: {}, accuracy: {}".format(index, curr_loss, curr_acc)
		print "[{}][{}/{}] loss={}, accuracy={}".format(epoch, index, int(len(train_dataset) / batch_size),curr_loss, curr_acc)

# After execution is done lets plot the values
pyplot.plot(np.array(loss),'b', label='loss')
pyplot.plot(np.array(accuracy),'r', label='accuracy')
pyplot.legend(loc='upper right')
pyplot.show()

##################################################################################
#### Save the trained model for testing later

# save as two protobuf files (predict_net.pb and init_net.pb)
# predict_net.pb defines the architecture of the network
# init_net.pb defines the network params/weights
print "Saving the trained model to predict/init.pb files..."
deploy_model = model_helper.ModelHelper(name="cnnm_deploy", arg_scope=arg_scope, init_params=False)
Add_CNN_M(deploy_model, "data")

# Use the MOBILE EXPORTER to save the deploy model as pbs
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter_test.py
workspace.RunNetOnce(deploy_model.param_init_net)
workspace.CreateNet(deploy_model.net, overwrite=True) # (?)
init_net, predict_net = mobile_exporter.Export(workspace, deploy_model.net, deploy_model.params)
with open(init_net_out, 'wb') as f:
    f.write(init_net.SerializeToString())
with open(predict_net_out, 'wb') as f:
    f.write(predict_net.SerializeToString())

print "Done, exiting..."





