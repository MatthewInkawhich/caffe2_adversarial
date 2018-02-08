# MatthewInkawhich
from __future__ import print_function
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
root_ucf_jpg_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/UCF11_updated_jpg_5FPS")
seq_size = 10
list_of_seqs = []
training_percent = .80
validation_percent = .10
testing_percent = .10
training_output = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'of_train_lmdb')
validation_output = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'of_validate_lmdb')
testing_output = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'of_test_lmdb')
classes = {"basketball":0,
            "biking":1,
            "diving":2,
            "golf_swing":3,
            "horse_riding":4,
            "soccer_juggling":5,
            "swing":6,
            "tennis_swing":7,
            "trampoline_jumping":8,
            "volleyball_spiking":9,
            "walking":10}


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


def count_labels(list_of_seqs, name):
    label_counts = np.zeros(len(classes))
    for seq in list_of_seqs:
        label = int(classes[seq[0].split('UCF11_updated_jpg_5FPS/')[1].split('/')[0]])
        label_counts[label] += 1
    print("\n*************\nLabel count for " + name)
    print("Total sequences: " + str(len(list_of_seqs)))
    for i, c in enumerate(label_counts):
        print(i, " :: ", int(c))
    print('\n')


def preprocess_and_create_lmdb(list_of_seqs, lmdb_name):
    print(">>> Write " + str(lmdb_name) + " database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY: just a very large number
    print("LMDB_MAP_SIZE", LMDB_MAP_SIZE)
    env = lmdb.open(lmdb_name, map_size=LMDB_MAP_SIZE)
    count = 0

    ### Preprocess sequences into (seq_size*2, image_height, image_width)
    for seq in list_of_seqs:
        label = int(classes[seq[0].split('UCF11_updated_jpg_5FPS/')[1].split('/')[0]])
        first = True
        for of_file in seq:
            of_img = cv2.imread(of_file).astype(np.float32)
            of_img = resize_image(of_img, 224, 224)
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
    print("\nLMDB saved at " + output)



########################################################################
# MAIN
########################################################################
########################################################################
# Generate list of contiguous sequences
########################################################################
# for each subdirectory in root dir [ex. UCF11_updated_mpg/basketball]
for dir1 in glob.glob(root_ucf_jpg_directory + '/*'):
    # for each subdirectory in new category level directory [ex. UCF11_updated_mpg/basketball/v_shooting_01]
    for dir2 in glob.glob(dir1 + "/*"):
        # for each level 3 directory [ex. UCF11_updated_mpg/basketball/v_shooting_01/01/]
        for dir3 in glob.glob(dir2 + "/*"):
            # get an array of jpgs in the directory (these are full paths)
            of_arr = glob.glob(dir3 + "/oflow_recentered/*.jpg")
            # sort array by h/v, then by sequence number
            of_arr.sort(key=lambda x: x.split('/')[-1].split('_f')[-1].split('_')[2].split('.')[0])
            of_arr.sort(key=lambda x: int(x.split('/')[-1].split('_f')[-1].split('_')[1]))
            # input all contiguous sequences of seq_size as lists into list_of_seqs list
            for i, path in enumerate(of_arr):
                if (i % 2 == 0):
                    if ((i + seq_size*2) <= (len(of_arr))):
                        list_of_seqs.append(of_arr[i:(i+seq_size*2)])

# randomly shuffle list of contiguous sequences
random.shuffle(list_of_seqs)

# print total number of sequences
num_sequences = len(list_of_seqs)
print('\nTotal number of sequences:', num_sequences)

# print label distribution over three lmdbs
num_training_sequences = int(num_sequences * training_percent)
num_validation_sequences = int(num_sequences * validation_percent)
num_testing_sequences = int(num_sequences * testing_percent)


count_labels(list_of_seqs[0:num_training_sequences], "training")
count_labels(list_of_seqs[num_training_sequences:num_training_sequences+num_validation_sequences], "validation")
count_labels(list_of_seqs[num_training_sequences+num_validation_sequences:num_training_sequences+num_validation_sequences+num_testing_sequences], "testing")


########################################################################
# Preprocess sequence and insert into LMDB
########################################################################
### Create testing lmdb
preprocess_and_create_lmdb(list_of_seqs[num_training_sequences+num_validation_sequences:num_training_sequences+num_validation_sequences+num_testing_sequences], testing_output)

### Create validation lmdb
preprocess_and_create_lmdb(list_of_seqs[num_training_sequences:num_training_sequences+num_validation_sequences], validation_output)

### Create training lmdb
preprocess_and_create_lmdb(list_of_seqs[0:num_training_sequences], training_output)
