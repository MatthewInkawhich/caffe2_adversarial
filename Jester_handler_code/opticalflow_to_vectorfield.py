# MatthewInkawhich
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'DukeML', 'caffe2_sandbox', 'lib'))
import image_manipulation
import optical_flow


########################################################################
# Configs
########################################################################
image_height = 100
image_width = 100
downsample = 5          # downsample param; show every <downsample> vector (1 == show all vectors)



########################################################################
# Functions
########################################################################

def print_all_optical_flow(h_oflow, v_oflow, f1, f2, downsample=1):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].set_title("frame 1")
    axarr[0,0].imshow(f1)
    axarr[0,1].set_title("frame 2")
    axarr[0,1].imshow(f2)
    axarr[1,0].set_title("horizontal flow")
    axarr[1,0].imshow(h_oflow, cmap='gray')
    axarr[1,1].set_title("vertical flow")
    axarr[1,1].imshow(v_oflow, cmap='gray')
    plt.tight_layout()
    X, Y, U, V = get_optical_flow_vector_field(h_oflow, v_oflow)
    print_vector_field(X, Y, U, V, "Optical Flow Vector Field", downsample)



########################################################################
# Main
########################################################################

### (1) Print optical flow from two sequential spatial images

# Paths to spatial jpg frames
im1 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00010.jpg')
im2 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00012.jpg')

# Calculate optical flow
h, v = optical_flow.calc_optical_flow(im1, im2, image_height, image_width)

# (or print just vector field)
X, Y, U, V = optical_flow.get_optical_flow_vector_field(h, v)
optical_flow.print_vector_field_with_diff(X, Y, U, V, "Plot title", [(40,15)], downsample)



### (2) Print vector field from optical flow jpgs (h and v)

# Paths to optical flow frames
h_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_h.jpg')
v_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_v.jpg')

# Preprocess optical flow from jpgs
h, v = optical_flow.process_optical_flow_jpg(h_path, v_path)
X, Y, U, V = optical_flow.get_optical_flow_vector_field(h, v)
optical_flow.print_vector_field(X, Y, U, V, "Plot title", downsample)
