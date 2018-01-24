# NAI
# This script takes a full dictionary as input
# The input dictionary is of the form: </path/to/jpg> <label>
# Once read in, we shuffle the lines then split them into two output
# files: a train dictionary and a test dictionary

import random
import os

input_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/FullDictionary_UCF11_updated_jpg_5FPS.txt")
output_train_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/TrainDictionary_UCF11_updated_jpg_5FPS.txt")
output_test_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/UCF11/TestDictionary_UCF11_updated_jpg_5FPS.txt")

# percentage of the file from the full dictionary that will go to train dictionary
percent_train = .8

# Read in all the lines from the full dictionary
lines = open(input_dict).readlines()

# Randomize the order of the lines
random.shuffle(lines)

# Decide how many will go to train
num_lines = len(lines)
num_train = int(num_lines*percent_train)

print "total lines read in: ", len(lines)
print "num train: ",len(lines[:num_train])
print "num test: ",len(lines[num_train:])
print "total lines written: ", (len(lines[:num_train])+len(lines[num_train:]))

# write the two separate files
open(output_train_dict, 'w').writelines(lines[:num_train])
open(output_test_dict, 'w').writelines(lines[num_train:])
