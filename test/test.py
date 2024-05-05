#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:44:38 2024

@author: joanneliang
"""
import cv2
import numpy as np

# Load the images
img = cv2.imread('app1.png')

# Convert it to GRAY
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Create window to display image
cv2.namedWindow('colorhist', cv2.WINDOW_AUTOSIZE)

#Create an empty image for the histogram
h = np.zeros((100,512))
bins = np.arange(64,dtype=np.int32).reshape(64,1)


# Calculate the histogram and normalize it
hist_img = cv2.calcHist([img_hsv], [0], None, [64], [1, 256])
cv2.normalize(hist_img, hist_img, 100, cv2.NORM_MINMAX);
hist=np.int32(np.around(hist_img))
pts = np.column_stack((bins,hist))

#Loop through each bin and plot the rectangle in white
for x,y in enumerate(hist):
    cv2.rectangle(h,(x*8,y),(x*8 + 8-1,100),(255),-1)

#Flip upside down
h=np.flipud(h)

#Show the histogram
cv2.imshow('colorhist',h)
cv2.waitKey(0)

