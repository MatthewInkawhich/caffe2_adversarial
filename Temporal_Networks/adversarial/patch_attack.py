##################################################################################
# NAI
#
# This file contains the Saliency Attack implementation for the CNNM network trained on
#   optical flow stacks and the process of reversing perturbations from an optical
#   flow stack back onto spatial images. 
# 
#   This file only considers a single optical flow stack. First, it verifies that
#   it is correctly classified, then it perturbs the stack with SA and makes 
#   sure it is misclassified, then it reverses the noise from the perturbed optical
#   flow back onto the images of the optical flow stack. Finally, it recomputes the 
#   optical flow between the images of the perturbed stack and verifies that the
#   optical flow stack is still adversarial.
#
##################################################################################

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
import cv2

##################################################################################
# Function to calculate dense optical flow between two adjacent frames
##################################################################################
def calc_optical_flow(frame1, frame2):

    # Convert the images to grayscale
    f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    # NATES CV2
    flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # MATTS CV2
    #flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Separate horizontal and vertical pieces
    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    # From beyond short snippits
    h_oflow[h_oflow < -40] = -40
    h_oflow[h_oflow > 40] = 40
    v_oflow[v_oflow < -40] = -40
    v_oflow[v_oflow > 40] = 40
    h_oflow = cv2.normalize(h_oflow, None, 0, 255, cv2.NORM_MINMAX)
    v_oflow = cv2.normalize(v_oflow, None, 0, 255, cv2.NORM_MINMAX)

    # Return the oflow
    return h_oflow,v_oflow


##################################################################################
### Get the original images given the optical flow image paths
##################################################################################
# oflow_seq = [\path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ...]
def get_images_from_oflow_seq( oflow_seq ):
    
    seq = oflow_seq[::2]
    #print seq

    prefix_path = os.path.dirname(seq[0]).replace("20bn-jester-v1-oflow","20bn-jester-v1")
    #print prefix_path 

    # List to store the filenames of the images we want, these are derived from the file names of the 
    #   optical flow paths that are input in oflow_seq
    # fnames = = [ '/full/path/to/00001.jpg', '/full/path/to/00002.jpg', '/full/path/to/00003.jpg', ...]
    fnames = []
    for file in seq:
        oflow_file = os.path.basename(file)
        i1 = prefix_path + "/" + oflow_file.split("_")[1] + ".jpg"
        i2 = prefix_path + "/" + oflow_file.split("_")[2] + ".jpg"
        if i1 not in fnames:
            fnames.append(i1)
        if i2 not in fnames:
            fnames.append(i2)

    assert(len(fnames) == 11)

    #images = np.zeros((11,3,100,100))
    images = np.zeros((11,100,100,3))
    for i,f in enumerate(fnames):
        # [0,255]
        of_img = cv2.imread(f).astype(np.float32)
        of_img = jdh.crop_center(of_img, 100, 100)
        #of_img = of_img.swapaxes(1, 2).swapaxes(0, 1)    # HWC to CHW dimension
        images[i,:,:,:] = of_img

    return images

##################################################################################
### Perturb the two spatial frames according to the perturbed oflow
##################################################################################
##################################################################################
# Function: perturb_images
#
# Inputs:
# - img1 - original frame 1
# - img2 - original frame 2
# - oflow_h - the perturbed horizontal optical flow component
# - oflow_v - the perturbed vertical optical flow component
# - perturb_loc - location of the perturbation to place on images
# - K - size of cluster to move
#
# Outputs:
# - adv_img1 - perturbed image (first in sequence)
# - adv_img2 - perturbed image (second in sequence)
#
def perturb_images(img1, img2, oflow_h, oflow_v, perturb_loc, K):

    #print img1.shape
    #print img1.max()
    #print img1.min()
    #print perturb_loc
    #exit()

    # Make copy of img1 and img2, which will be perturbed and returned
    adv1 = np.copy(img1)
    adv2 = np.copy(img2)

    # Location to perturb in the first image
    # x,y = loc
    row = perturb_loc[0]
    col = perturb_loc[1]

    # Location to perturb in the second image
    #row2 = row - 3*oflow_v[row,col]
    #col2 = col - 3*oflow_h[row,col]

    # compute the trend of motion in the neighborhood of the location
    # Size of neighborhood
    neigh = 8
    cnt = 0

    # Calc the average of the horizontal components in the neighborhood
    avg_h = 0
    for i in range(neigh):
        for j in range(neigh):
            if ((int(row+i) > 0) and (int(row+i) < 100) and (int(col+j) > 0) and (int(col+j) < 100)):
                avg_h += oflow_h[int(row+i),int(col+j)]
                cnt += 1
    avg_h /= cnt

    # Calc the average of the vertical components in the neighborhood
    avg_v = 0
    cnt = 0
    for i in range(neigh):
        for j in range(neigh):
            if ((int(row+i) > 0) and (int(row+i) < 100) and (int(col+j) > 0) and (int(col+j) < 100)):
                avg_v += oflow_v[int(row+i),int(col+j)]
                cnt += 1
    avg_v /= cnt

    # Set the new location to be motion in the opposite direction of the dominant movement in the neighborhood
    row2 = int(row + 10*avg_v)
    col2 = int(col + 10*avg_h)


    print "loc: ",loc
    print "row: ",row
    print "col: ",col
    print "row2: ",row2
    print "col2: ",col2

    # TODO: Bounds Checking!!!!!!
    # Make sure perturbing at row,col in img1 will not cause us to go out of bounds
    # Make sure perturbing the second image at [(row,col) + oflow(row,col)] will not be out of bounds
    if ((row2 > 0) and (row2 < 100) and (col2 > 0) and (col2 < 100)): 
        # Place cluster of K pixels with color C onto img1 at loc
        for i in range(K):
            for j in range(K):
                if ((int(row+i) > 0) and (int(row+i) < 100) and (int(col+j) > 0) and (int(col+j) < 100)):
                    adv1[int(row+i),int(col+j)] = img1[row,col]

        # Place cluster of K pixels with color C onto img2 at loc( row + oflow_v(row,col), col + oflow_h(row,col) )
        for i in range(K):
            for j in range(K):
                if ((int(row2+i) > 0) and (int(row2+i) < 100) and (int(col2+j) > 0) and (int(col2+j) < 100)):
                    adv2[int(row2+i),int(col2+j)] = img1[row,col]

    # Return adversarial images
    return adv1,adv2

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################

##################################################################################
### Handle inputs and configs
##################################################################################
N = 100 # Number of locations in the stack to perturb
K = 10
#TEST_DICT = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'SingleVideoTestDictionary.txt')
TEST_DICT = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'VerySmallTestDictionary_5class.txt')
INIT_NET = os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'Temporal_Networks', 'CNNM_4epoch_jester_init_net.pb')
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
test_model.param_init_net = tmp_param_net

# Populate the model obj with the predict net stuff, which defines the structure of the model
predict_net_proto = caffe2_pb2.NetDef()
with open(PREDICT_NET, "rb") as f:
    predict_net_proto.ParseFromString(f.read())
tmp_predict_net = core.Net(predict_net_proto)
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
for image, label, seq in test_dataset.read(batch_size=1,shuffle=False):
    workspace.FeedBlob("data", image)
    workspace.FeedBlob("label", label)
    break

# Run a test pass on the test net
workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net, overwrite=True)

# Keeping stats
num_correct = 0
num_succeed = 0
num_attacks = 0
total = 0

for stack, label, seq in test_dataset.read(batch_size=1,shuffle=False):

    total += 1

    print "stack,",stack.min()
    print stack.max()

    # Run the stack through the network which will produce a softmax vector and populate the gradient blobs
    # stack.shape = [1, 20, 100, 100]
    workspace.FeedBlob("data", stack)
    workspace.FeedBlob("label", label)
    workspace.RunNet(test_model.net)
    
    # Fetch softmax from workspace to see what the original prediction is
    softmax = workspace.FetchBlob('softmax')
    # Get the top-1 prediction
    max_index, max_value = max(enumerate(softmax[0]), key=operator.itemgetter(1))

    print ("Softmax = ",softmax)
    print ("label = ",label)
    print ("prediction = ",max_index)

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
    print("stack.shape", stack.shape)

    # Take the element-wise absolute value of the gradient of loss w.r.t data to
    #     approximate the saliency map [https://arxiv.org/pdf/1312.6034.pdf]
    saliency_map = np.absolute(data_grad)
    saliency_map = saliency_map[0]

    # Find the coordinates of the N largest (i.e. most influential/salient features)
    # These are 3D coords [c, h, w]
    # b is always 0 since we are not feeding in test batches
    # locs = [[c, h, w], [c, h, w], [c, h, w], ...]
    indices =  np.argpartition(saliency_map.flatten(), -N)[-N:]
    locs =  np.vstack(np.unravel_index(indices, saliency_map.shape)).T

    # Get the images corresponding to this stack
    # These are the original images that were used to calculate the optical flows
    # Seq = [\path\to\0_h.jpg, \path\to\0_v.jpg, \path\to\1_h.jpg, \path\to\1_v.jpg, ...]
    # Img_arr.shape = (11,100,100,3)
    Img_arr = get_images_from_oflow_seq(seq[0])
    print Img_arr.shape

    # For each of the N most salient locations
    for loc in locs:
        print "Perturbing stack at: ",loc
        # Get the depth lined up on a h oflow slice
        if (loc[0]%2 != 0):
            loc[0] -= 1

        # Find the index of the images to perturb in img_arr based on loc
        ind = loc[0] // 2
        # Perturb the two relevant images
        print "perturb_images(Img_arr[ {} ], Img_arr[ {} ], stack[ {} ], stack[ {} ], {}, {})".format(ind,ind+1, loc[0], loc[0]+1, [loc[1],loc[2]] , K)
        Img_arr[ ind ],Img_arr[ ind+1 ] = perturb_images(Img_arr[ ind ], Img_arr[ ind+1 ], stack[0][ loc[0] ], stack[0][ loc[0]+1 ], [loc[1],loc[2]], K)


    ##############################################################
    # Recompute optical flow between perturbed spatial frames
    ##############################################################
    # Allocate space to put newly calculate optical flow
    new_pstack = np.zeros((1,20,100,100))

    # For every pair of images, calculate optical flow and store it in the new perturbed stack
    cnt = 0
    for i in range(0,10):
        #print "oflow between {} , {}".format(i,i+1)
        #print "adding to {} , {}".format(cnt,cnt+1)
        tmph,tmpv = calc_optical_flow(Img_arr[i].astype(np.uint16), Img_arr[i+1].astype(np.uint16))
        new_pstack[0,cnt] = tmph
        new_pstack[0,cnt+1] = tmpv
        cnt += 2   

    new_pstack /= 255
    new_pstack[ new_pstack<0 ] = 0
    new_pstack[ new_pstack>1 ] = 1
    print "new pstack:",new_pstack.min()
    print new_pstack.max()

    ##############################################################
    # Check to see if the recomputed oflow is still adversarial
    ##############################################################
    # Run the stack through the network which will produce a softmax vector and populate the gradient blobs
    # stack.shape = [1, 20, 100, 100]
    workspace.FeedBlob("data", new_pstack.astype(np.float32))
    workspace.FeedBlob("label", label)
    workspace.RunNet(test_model.net)
    
    # Fetch softmax from workspace to see what the original prediction is
    softmax = workspace.FetchBlob('softmax')
    # Get the top-1 prediction
    max_index, max_value = max(enumerate(softmax[0]), key=operator.itemgetter(1))

    print ("Softmax = ",softmax)
    print ("label = ",label)
    print ("prediction = ",max_index)

    ##### If initial prediction is incorrect, no need to attack
    if (max_index != label[0]):
        print "SUCCESSFUL ADVERSARIAL IMAGE" 
        num_succeed += 1
    else:
        print "FAILED ADVERSARIAL IMAGE"
        num_correct += 1

    #exit()

    ##############################################################
    # Visualize the perturbed spatial frames
    ##############################################################
    
    Img_arr /= 255.

    plt.subplot(3,4,1)
    plt.axis("off")
    new = Img_arr[0]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,2)
    plt.axis("off")
    new = Img_arr[1]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,3)
    plt.axis("off")
    new = Img_arr[2]
    plt.imshow(new[:,:,(2,1,0)])

    plt.subplot(3,4,4)
    plt.axis("off")
    new = Img_arr[3]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,5)
    plt.axis("off")
    new = Img_arr[4]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,6)
    plt.axis("off")
    new = Img_arr[5]
    plt.imshow(new[:,:,(2,1,0)])

    plt.subplot(3,4,7)
    plt.axis("off")
    new = Img_arr[6]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,8)
    plt.axis("off")
    new = Img_arr[7]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,9)
    plt.axis("off")
    new = Img_arr[8]
    plt.imshow(new[:,:,(2,1,0)])

    plt.subplot(3,4,10)
    plt.axis("off")
    new = Img_arr[9]
    plt.imshow(new[:,:,(2,1,0)])
    plt.subplot(3,4,11)
    plt.axis("off")
    new = Img_arr[10]
    plt.imshow(new[:,:,(2,1,0)])
    plt.show()
    
    exit()
    
    '''
    plt.subplot(1,4,1)
    plt.title("Orig oflow")
    plt.imshow(stack[0,0,:,:],cmap='gray')
    plt.subplot(1,4,2)
    plt.title("Data Grad")
    plt.imshow(data_grad[0,1,:,:],cmap='gray')
    plt.subplot(1,4,3)
    plt.title("Noise field")
    plt.imshow(epsilon_sign_matrix[0,1,:,:],cmap='gray')
    plt.subplot(1,4,4)
    plt.title("perturbed oflow")
    plt.imshow(pstack[0,0,:,:],cmap='gray')
    plt.show()

    exit()
    '''


print ("\n**************************************")
#print ("Epsilon = ",epsilon)
print ("Total Stacks Tested = ",total)
print ("Number of attacks attempted = ",num_attacks)
print ("Number of successful attacks = ",num_succeed)
print ("Old Accuracy = ",num_attacks/float(total))
print ("Adv Accuracy = ",num_correct/float(total))


