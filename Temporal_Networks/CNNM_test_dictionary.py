##################################################################################
# NAI
#
# This script is what we will use for testing our trained model based on a labeled
#   test dictionary. This does not format a csv for submission but can be used to
#   quickly see how the model we just trained does on our test set. Note, the test
#   dictionary is labeled, so this will output a % ACCURACY
#
##################################################################################

# import dependencies
print "Import Dependencies..."
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import operator
import caffe2.python.predictor.predictor_exporter as pe
from caffe2.python import core,model_helper,net_drawer,optimizer,workspace,visualize,brew,utils
from caffe2.proto import caffe2_pb2
from caffe2.python.predictor import mobile_exporter
import random
import skimage.io
from skimage.color import rgb2gray
import JesterDatasetHandler as jdh

##################################################################################
# Gather Inputs
test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TestDictionary_5class.txt")
# predict_net = "CNNM_jester_predict_net.pb"
# init_net = "CNNM_4epoch_jester_init_net.pb"

init_net = os.path.join(os.path.expanduser('~'),"DukeML", "models", "CNNM_3", "CNNM_3epoch_jester_init_net.pb")
predict_net = os.path.join(os.path.expanduser('~'),"DukeML", "models", "CNNM_3", "CNNM_jester_predict_net.pb")


##################################################################################
# Bring up the network from the .pb files
with open(init_net) as f:
    init_net = f.read()
with open(predict_net) as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)


##################################################################################
# Loop through the test dictionary and run the inferences

# Confusion Matrix
cmat = np.zeros((5,5))

# Initialization.
test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)

num_correct = 0
total = 0

# Cycle through the test dictionary once, with batch size = 1, meaning we only consider one stack at a time
for stack, label in test_dataset.read(batch_size=1):

    # Run the stack through the predictor and get the result array
    results = np.asarray(p.run([stack]))[0,0,:]
    print results
    # Get the top-1 prediction
    max_index, max_value = max(enumerate(results), key=operator.itemgetter(1))

    print "Prediction: ", max_index
    print "Confidence: ", max_value

    # Update confusion matrix
    cmat[label,max_index] += 1

    if max_index == label:
        num_correct += 1

    total += 1

print "\n**************************************"
print "Total Tests = {}".format(total)
print "# Correct = {}".format(num_correct)
print "Accuracy = {}".format(num_correct/float(total))

# Plot confusion matrix
fig = plt.figure()
#plt.clf()
plt.tight_layout()
ax = fig.add_subplot(111)
#ax.set_aspect(1)
res = ax.imshow(cmat, cmap=plt.cm.rainbow,interpolation='nearest')

width, height = cmat.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cmat[x,y]), xy=(y, x),horizontalalignment='center',verticalalignment='center')

#cb = fig.colorbar(res)
classes = ['Swipe Left', 'Swipe Right', 'Swipe Down', 'Swipe Up', 'Push Away']
plt.xticks(range(width), classes, rotation=17)
plt.yticks(range(height), classes, rotation=17)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
plt.title('Jester 5-class Confusion Matrix')
plt.show()
