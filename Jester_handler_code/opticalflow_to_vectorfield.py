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
    X, Y, U, V = optical_flow.get_optical_flow_vector_field(h_oflow, v_oflow)
    optical_flow.print_vector_field(X, Y, U, V, "Optical Flow Vector Field", downsample)



########################################################################
# Main
########################################################################

### (1) Print optical flow from two sequential spatial images

#Paths to spatial jpg frames
im1 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00010.jpg')
im2 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00011.jpg')
# im1_p = os.path.join(os.path.expanduser('~'), 'image_manipulation', 'images', '00013__noise0.05.jpg')
# im2_p = os.path.join(os.path.expanduser('~'), 'image_manipulation', 'images', '00014__noise0.05.jpg')
im1_p = os.path.join(os.path.expanduser('~'), 'image_manipulation', 'images', '00013.jpg')
im2_p = os.path.join(os.path.expanduser('~'), 'image_manipulation', 'images', '00014.jpg')


i1 = cv2.imread(im1_p)
i1 = i1[:, :, (2, 1, 0)]
i2 = cv2.imread(im2_p)
i2 = i2[:, :, (2, 1, 0)]

n=10
for r in range(83,83+n):
    for c in range(15,15+n):
        i1[r,c] = (0,255,0)

for r in range(83,83+n):
    for c in range(0,0+n):
        i2[r,c] = (0,255,0)


# cv2.imwrite(im1_p, i1)
# cv2.imwrite(im2_p, i2)

# Calculate optical flow
h, v = optical_flow.calc_optical_flow(im1_p, im2_p, image_height, image_width)

# (or print just vector field)
X, Y, U, V = optical_flow.get_optical_flow_vector_field(h, v)
#optical_flow.print_vector_field_with_diff(X, Y, U, V, "Plot title", [(40,15)], downsample)

print_all_optical_flow(h, v, i1, i2, downsample=1)
#
# plt.imshow(h, cmap='gray')
# plt.show()
# plt.imshow(v, cmap='gray')
# plt.show()

# ### (2) Print vector field from optical flow jpgs (h and v)
#
# # Paths to optical flow frames
# h_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_h.jpg')
# v_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_v.jpg')
#
# # Preprocess optical flow from jpgs
# h, v = optical_flow.process_optical_flow_jpg(h_path, v_path)
# X, Y, U, V = optical_flow.get_optical_flow_vector_field(h, v)
# optical_flow.print_vector_field(X, Y, U, V, "Plot title", downsample)

# im1_p = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'f1.jpg')
# im2_p = os.path.join(os.path.expanduser('~'), 'DukeML', 'junk', 'f2.jpg')
#
# im1 = np.zeros((100, 100))
# im2 = np.zeros((100, 100))
#
# for r in range(20,40):
#     for c in range(20,40):
#         im1[r,c] = 255
# im1[60,60] = 255
# im2[65,65] = 255
#
# for r in range(25,45):
#     for c in range(20,40):
#         im2[r,c] = 255
#
# cv2.imwrite(im1_p, im1)
# cv2.imwrite(im2_p, im2)
# # f, axarr = plt.subplots(1,2)
# # axarr[0].imshow(im1)
# # axarr[1].imshow(im2)
# # plt.show()
#
# h, v = optical_flow.calc_optical_flow(im1_p, im2_p, 100, 100)
# X, Y, U, V = optical_flow.get_optical_flow_vector_field(h, v)
# print_all_optical_flow(h, v, im1, im2, downsample=5)
