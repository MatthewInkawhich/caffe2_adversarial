import os
import numpy as np
from scipy.misc import imresize, imsave
import cv2


# Returns a new numpy matrix center cropped to specified height and width
def crop_center(img, new_height, new_width):
    orig_height, orig_width, _ = img.shape
    startx = (orig_width//2) - (new_width//2)
    starty = (orig_height//2) - (new_height//2)
    return img[starty:starty+new_height, startx:startx+new_width]


# If the desired height or width is less than the current height or width, return imresize output,
#   else, use crop_center to resize image
def resize_image(img, new_height, new_width):
    h, w, _ = img.shape
    if (h < new_height or w < new_width):
        img_data_r = imresize(img, (new_height, new_width))
    else:
        img_data_r = crop_center(img, new_height, new_width)
    return img_data_r


# Returns image the same as the input image, but instead of (x,y,3), it is (x,y,1)
def handle_greyscale(img):
    img = img[:,:,0]
    img = np.expand_dims(img, axis=2)
    return img
