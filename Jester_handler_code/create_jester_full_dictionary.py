# NAI

import os
import glob

# ***************************************************************
# function to read the key file and return a dictionary that maps english categories to int labels
def read_key_file(key_file):
    # this function reads the keyfile and returns a map whose keys are
    #	the categories in english and whose value is the int label
    key_dict = {}

    f = open(key_file,"r")
    for line in f:
        dat = line.split(",")
        english = dat[1].rstrip()
        key_dict[english] = dat[0]
    f.close()

    return key_dict

# ***************************************************************
# MAIN

# path to root of jester directory
root_jester_directory = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/20bn-jester-v1")
# path to key file
jester_key_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/KEYFILE_jester.txt")
# path to jester train label csv
jester_train_csv = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/jester-v1-train.csv")
# name of output file to save dictionary
ofile_name = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/FullDictionary.txt")

fin = open(jester_train_csv, "rb")
f = open(ofile_name, 'wb')

# Get the dictionary of key value pairs from the key file
key_dict = read_key_file(jester_key_file)

for line in fin:

    lsplit = line.split(";")
    dir = lsplit[0]
    label = lsplit[1].rstrip()
    int_label = key_dict[label]
    print "{} : {} = {}".format(dir,label,int_label)
    ostring = "{}/{} {}".format(root_jester_directory,dir,int_label)
    print ostring
    f.write(ostring + "\n")


f.close()
