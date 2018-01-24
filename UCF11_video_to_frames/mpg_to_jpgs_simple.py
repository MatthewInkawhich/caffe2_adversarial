# Trying to decompose a video to frames and save each frame as a .jpg
# https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import cv2

vid = "samples/v_golf_01_01.mpg"

vidcap = cv2.VideoCapture(vid)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
