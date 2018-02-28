import os
import numpy as np
import cv2
import image_manipulation
import matplotlib.pyplot as plt


# Takes two adjacent image paths and desired height and width, and returns horizontal
#   and vertical optical flow components
def calc_optical_flow(im1, im2, image_height, image_width):
    # calculate dense optical flow
    # settings from tutorial
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
    cv2_version = int(cv2.__version__.split('.')[0])
    frame1 = image_manipulation.resize_image(cv2.imread(im1), image_height, image_width)
    f1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = image_manipulation.resize_image(cv2.imread(im2), image_height, image_width)
    f2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if cv2_version > 2:
        # NATES CV2
        flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        # MATTS CV2
        flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 30, 3, 5, 1.2, 0)

    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    print "\tBefore adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    # h_oflow[h_oflow < -127] = -127
    # h_oflow[h_oflow > 127] = 127
    # v_oflow[v_oflow < -127] = -127
    # v_oflow[v_oflow > 127] = 127
    # h_oflow = np.rint(h_oflow)
    # v_oflow = np.rint(v_oflow)
    h_oflow *= 10
    v_oflow *= 10
    #h_oflow += 127
    #v_oflow += 127
    #h_oflow[h_oflow > 255] = 255
    #h_oflow[h_oflow < 0] = 0
    #v_oflow[v_oflow > 255] = 255
    #v_oflow[v_oflow < 0] = 0
    #h_oflow = np.rint(h_oflow)
    #_oflow = np.rint(v_oflow)
    print "\tAfter adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    return h_oflow, v_oflow



def write_optical_flow(img1, img2, ofile_name_horizontal, ofile_name_vertical, image_height, image_width):
    cv2_version = int(cv2.__version__.split('.')[0])

    frame1 = image_manipulation.crop_center(cv2.imread(img1), image_height, image_width)
    frame2 = image_manipulation.crop_center(cv2.imread(img2), image_height, image_width)
    f1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    f2_gray = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # calculate dense optical flow
    # settings from tutorial
    # https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

    if cv2_version > 2:
        # NATES CV2
        flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    else:
        # MATTS CV2
        flow = cv2.calcOpticalFlowFarneback(f1_gray,f2_gray, 0.5, 3, 30, 3, 5, 1.2, 0)

    h_oflow = flow[...,0]
    v_oflow = flow[...,1]

    print "\tBefore adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()


    # Normalize
    # Multiply by 10, recenter to 127, cap at [0,255], round to int
    # h_oflow *= 10
    # v_oflow *= 10
    # h_oflow += 127
    # v_oflow += 127
    # h_oflow[h_oflow > 255] = 255
    # h_oflow[h_oflow < 0] = 0
    # v_oflow[v_oflow > 255] = 255
    # v_oflow[v_oflow < 0] = 0

    
    # Clip and norm
    h_oflow[h_oflow < -10] = -10
    h_oflow[h_oflow > 10] = 10
    v_oflow[v_oflow < -10] = -10
    v_oflow[v_oflow > 10] = 10
    h_oflow = cv2.normalize(h_oflow, None, 0, 255, cv2.NORM_MINMAX)
    v_oflow = cv2.normalize(v_oflow, None, 0, 255, cv2.NORM_MINMAX)
    #h_oflow = np.rint(h_oflow)
    #v_oflow = np.rint(v_oflow)
    
    '''
    # Clip based on evaluation of distribution
    # This range should include about 99.9% of values
    h_oflow[h_oflow < -8] = -8
    h_oflow[h_oflow > 8] = 8
    v_oflow[v_oflow < -8] = -8
    v_oflow[v_oflow > 8] = 8
    # Scale the space [-10,10] to [0,255]
    OldMax = 8
    OldMin = -8
    NewMax = 255
    NewMin = 0
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
    #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    h_oflow = (((h_oflow - OldMin) * NewRange) / OldRange) + NewMin
    v_oflow = (((v_oflow - OldMin) * NewRange) / OldRange) + NewMin
    '''


    print "\tAfter adjustment..."
    print "\tmax: ",h_oflow.max()
    print "\tmin: ",h_oflow.min()
    print "\tmean: ",h_oflow.mean()

    # Save the optical flow displacement fields as images
    #cv2.imwrite(ofile_name_horizontal, normed_h)
    #cv2.imwrite(ofile_name_vertical, normed_v)

    cv2.imwrite(ofile_name_horizontal, h_oflow)
    cv2.imwrite(ofile_name_vertical, v_oflow)


# Returns optical flow vector field as 4 matrices to be fed into the
#   print_vector_field functions
def get_optical_flow_vector_field(h_oflow, v_oflow):
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
                U[i,j] = h_oflow[i,j]
                V[i,j] = v_oflow[i,j]
    return X, Y, U, V



# Print vector field data from get_optical_flow_vector_field function
def print_vector_field(X, Y, U, V, title, downsample=1):
    # Plot optical flow flow field
    row, col = X.shape
    for i in range(row):
        for j in range(col):
            if i % downsample != 0 or j % downsample != 0:
                U[i,j] = 0
                V[i,j] = 0
    plt.figure()
    plt.title(title)
    plt.quiver(X, Y, U, -V, scale=1, units='xy')
    plt.xlim(-1, col)
    plt.ylim(-1, row)
    plt.gca().invert_yaxis()
    plt.show()



# Print vector field data from get_optical_flow_vector_field function, but also
#   takes a list of (row, col) coordinates of where to print red arrows instead of black
def print_vector_field_with_diff(X, Y, U, V, title, diff=[], downsample=1):
    # Plot optical flow flow field
    row,col = X.shape
    for i in range(row):
        for j in range(col):
            if i % downsample != 0 or j % downsample != 0:
                U[i,j] = 0
                V[i,j] = 0
    C = np.zeros((row,col))
    for r, c in diff:
        C[r,c] = 1
    plt.figure()
    plt.title(title)
    plt.quiver(X, Y, U, -V, C, scale=1, units='xy', cmap='bwr')
    plt.xlim(-1, col)
    plt.ylim(-1, row)
    plt.gca().invert_yaxis()
    plt.show()



# Takes horizontal and vertical oflow jpgs and recenters data back to zero
def process_optical_flow_jpg(h_path, v_path):
    h_img = cv2.imread(h_path)
    v_img = cv2.imread(v_path)
    h_img = h_img[:,:,0]
    v_img = v_img[:,:,0]
    # Recenter from 127 back to 0
    h_img = h_img.astype(int) - 127
    v_img = v_img.astype(int) - 127
    return h_img, v_img
