# NAI
# This script prints the contents of a pb file to stdout.
# It takes one cmd line arg which is the path to the .pb file to be printed
# This may be helpful for finetuning because you can see what the names of the ops in the model

# ex. python print_pb.py ~/DukeML/models/squeezenet/predict_net.pb

# import dependencies
print "Import Dependencies..."
import sys
import os
from caffe2.python import core,model_helper,workspace,brew,utils
from caffe2.proto import caffe2_pb2

if len(sys.argv) != 2:
	print "Usage: print_pb.py </path/to/file.pb>"
	exit()
else:
	pb_loc = sys.argv[1]

# Make sure the specified input exist
if not os.path.exists(pb_loc):
	print "ERROR: The input pb was not found"
	exit()

# construct a net object from the input pb file
net_proto = caffe2_pb2.NetDef()
with open(pb_loc, "rb") as f:
    net_proto.ParseFromString(f.read())
pred_net = core.Net(net_proto)

# print the net's prototxt to stdout
#print str(pred_net.Proto())

for i,op in enumerate(net_proto.op):
	print "\n******************************"
	print "OP: ", i
	print "******************************"
	print "OP_NAME: ",op.name
	print "OP_INPUT: ",op.input 
	print "OP_OUTPUT: ",op.output
	print "OP_SHAPE: ",op.arg[0]





