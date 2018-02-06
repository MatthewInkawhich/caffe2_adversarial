# NAI


import random
import os

input_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/FullDictionary.txt")
output_train_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TrainDictionary_5class.txt")
output_test_dict = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TestDictionary_5class.txt")
# percentage of the file from the full dictionary that will go to train dictionary
percent_train = .8

# Read in all the lines from the full dictionary
lines = open(input_dict).readlines()

##########################################################################
# Do you only want to consider some classes?
# Comment this out if you want all classes
desired_classes = [0,1,2,3,4] # Only keep the lines with these classes
new_lines = []
for line in lines:
    if int(line.split()[1].rstrip()) in desired_classes:
        new_lines.append(line)
        print line.rstrip()
lines=new_lines
##########################################################################

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
