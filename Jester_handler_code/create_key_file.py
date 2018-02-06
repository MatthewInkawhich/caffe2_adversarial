# NAI

import os
import glob

# Label file from download
label_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/jester-v1-labels.csv")

# where to save key file
key_file = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/KEYFILE_jester.txt")

# open label file for reading
fin = open(label_file, "rb")
fout = open(key_file, "w")

for i,line in enumerate(fin):
    ostring = "{},{}".format(i,line.rstrip())
    print ostring
    fout.write(ostring + "\n")



