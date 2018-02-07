# NAI
# THIS IS CUSTOM FOR JESTER DATASET

# MatthewInkawhich
#from __future__ import print_function
import numpy as np
import os
import glob
import random
import cv2
from scipy.misc import imresize, imsave
import lmdb
from caffe2.proto import caffe2_pb2



########################################################################
# Configs
########################################################################
# Input dictionary (assuming this is not the full dictionary)
input_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TrainDictionary_5class.txt")
seq_size = 10
list_of_seqs = []
training_output = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', 'jester_oflow_10stacked_train_lmdb')




########################################################################
# Functions
########################################################################
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


def handle_greyscale(img):
    img = img[:,:,0]
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_and_create_lmdb(list_of_seqs, lmdb_name):
    print(">>> Write " + str(lmdb_name) + " database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY: just a very large number
    print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
    env = lmdb.open(lmdb_name, map_size=LMDB_MAP_SIZE)
    count = 0

    ### Preprocess sequences into (seq_size*2, image_height, image_width)
    for s in list_of_seqs:

        seq = s[0]
        label = int(s[1])
        
        #exit()

        first = True
        for of_file in seq:
            of_img = cv2.imread(of_file).astype(np.float32)
            of_img = resize_image(of_img, 100, 100)
            of_img = handle_greyscale(of_img)
            if (first):
                seq_for_lmdb = of_img
                first = False
            else:
                seq_for_lmdb = np.append(seq_for_lmdb, of_img, axis=0)
        #seq_for_lmdb now contains numpy "image" with seq_size*2 channels

    ### Load sequence into LMDB
        with env.begin(write=True) as txn:

            def insert_image_to_lmdb(img_data, label, index):
                # Create TensorProtos
                tensor_protos = caffe2_pb2.TensorProtos()
                img_tensor = tensor_protos.protos.add()
                img_tensor.dims.extend(img_data.shape)
                img_tensor.data_type = 1
                flatten_img = img_data.reshape(np.prod(img_data.shape))
                img_tensor.float_data.extend(flatten_img)
                label_tensor = tensor_protos.protos.add()
                label_tensor.data_type = 2
                label_tensor.int32_data.append(label)
                txn.put(
                    '{}'.format(index).encode('ascii'),
                    tensor_protos.SerializeToString()
                )
                if ((index % 100 == 0)):
                    print("Inserted {} rows".format(index))
                index = index + 1
                return index

            count = insert_image_to_lmdb(seq_for_lmdb,label,count)
    print("Inserted {} rows".format(count))
    print("\nLMDB saved at " + training_output)



########################################################################
# MAIN
########################################################################

# Open the input dictionary file for reading
# Each line contains a path to a directory full of jpgs that correspond
#   to ONE video and the integer label representing the class for that video
infile = open(input_dictionary,"rb")

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
        list_of_seqs.append([single_seq, label])

# randomly shuffle list of contiguous sequences
random.shuffle(list_of_seqs)

# print total number of sequences
num_sequences = len(list_of_seqs)
print('\nTotal number of sequences:', num_sequences)

########################################################################
# Preprocess sequence and insert into LMDB
########################################################################

### Create training lmdb
preprocess_and_create_lmdb(list_of_seqs, training_output)


