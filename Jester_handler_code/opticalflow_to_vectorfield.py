# MatthewInkawhich
import numpy as np
np.set_printoptions(threshold=np.nan)
import os
import cv2
import matplotlib.pyplot as plt



########################################################################
# Configs
########################################################################
image_height = 100
image_width = 100
downsample = 5          # downsample param; show every <downsample> vector (1 == show all vectors)



########################################################################
# Functions
########################################################################
def crop_center(img, new_height, new_width):
    orig_height, orig_width, _ = img.shape
    startx = (orig_width//2) - (new_width//2)
    starty = (orig_height//2) - (new_height//2)
    return img[starty:starty+new_height, startx:startx+new_width]

def load_and_preprocess_opticalflow_jpg(path, image_height, image_width):
    img = cv2.imread(path)
    img = crop_center(img, image_height, image_width)
    img = img[:,:,0]
    img = cv2.normalize(img, None, -40, 40, cv2.NORM_MINMAX)
    return img

def calc_optical_flow(im1, im2):
    # calculate dense optical flow
    # settings from tutorial
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    frame1 = crop_center(cv2.imread(im1), image_height, image_width)
    f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = crop_center(cv2.imread(im2), image_height, image_width)
    f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # NATES CV2
    #flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # MATTS CV2
    flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 15, 3, 5, 1.2, 0)
    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    print "\tBefore adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    h_oflow[h_oflow < -127] = -127
    h_oflow[h_oflow > 127] = 127
    v_oflow[v_oflow < -127] = -127
    v_oflow[v_oflow > 127] = 127
    h_oflow = np.rint(h_oflow)
    v_oflow = np.rint(v_oflow)

    print "\tAfter adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    return h_oflow, v_oflow, f1_gray, f2_gray


def print_vector_field(h_oflow, v_oflow, downsample=1):
    # X: x-coordinates of the arrow tails
    # Y: y-coordinates of the arrow tails
    # U: x components of arrow vectors
    # V: y components of arrow vectors
    image_height, image_width = h_oflow.shape

    # Create meshgrid for coordinate resemblance
    Y, X = np.mgrid[0:image_height, 0:image_width]

    # Initialize U and V as the same size as image
    U = np.zeros((image_height,image_width))
    V = np.zeros((image_height,image_width))

    # Fill U and V with x and y vector components from h_img and v_img
    for i in range(image_height):
            for j in range (image_width):
                if i%downsample==0 and j%downsample==0:
                    U[i,j] = h_oflow[i,j]
                    V[i,j] = v_oflow[i,j]

    # Plot optical flow flow field
    plt.figure()
    plt.title("Optical Flow Vector Field")
    plt.quiver(X, Y, U, V, scale=1, units='xy')
    plt.xlim(-1, image_width)
    plt.ylim(-1, image_height)
    plt.gca().invert_yaxis()
    plt.show()


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
    print_vector_field(h_oflow, v_oflow, downsample)


def preprocess_optical_flow_jpg(h_path, v_path):
    h_img = cv2.imread(h_path)
    v_img = cv2.imread(v_path)
    h_img = h_img[:,:,0]
    v_img = v_img[:,:,0]
    # Recenter from 127 back to 0
    h_img = h_img.astype(int) - 127
    v_img = v_img.astype(int) - 127
    return h_img, v_img


########################################################################
# Main
########################################################################


### (1) Print optical flow from two sequential spatial images

# Paths to spatial jpg frames
im1 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00010.jpg')
im2 = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1', '13377', '00012.jpg')

# Calculate optical flow
h, v, f1_gray, f2_gray = calc_optical_flow(im1, im2)

# Print optical flow
print_all_optical_flow(h, v, f1_gray, f2_gray, downsample)

# (or print just vector field)
#print_vector_field(h, v, downsample)



# ### (2) Print vector field from optical flow jpgs (h and v)
#
# # Paths to optical flow frames
# h_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_h.jpg')
# v_path = os.path.join(os.path.expanduser('~'), 'DukeML', 'datasets', 'jester', '20bn-jester-v1-oflow', '13377', 'oflow_00010_00012_4_v.jpg')
#
# # Preprocess optical flow from jpgs
# h, v = preprocess_optical_flow_jpg(h_path, v_path)
#
# print_vector_field(h, v, downsample)
