# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:11:28 2019

@author: Ankush Jain
"""
import numpy as np
import matplotlib.pyplot as plt
from math import floor, degrees, atan2
from cv2 import imread, imshow, waitKey

# Function for performing Convolution
def convolution(m1, m2, maskSum):
    n, m = m1.shape
    #print(m1,m2,maskSum)
    mSum = 0
    for i in range(n):
        for j in range(m):
            mSum += (int(m1[i, j]) * int(m2[i, j]))
    return int(round(mSum/maskSum))

#
def gradient_mag_and_angle(gradX, gradY, gradientSize, gaussianSize):
    # Setting mask parameters
    outMag = np.zeros(gradX.shape, dtype='uint8')
    outAng = np.zeros(gradX.shape, dtype='float')
    # Variable to hold undefined area from gaussian filtering
    undefPixels = floor(gaussianSize/2) + floor(gradientSize/2)
    # Getting img dimensions
    n, m = gradX.shape
    # Starting gaussian masking
    for i in range(undefPixels, n-undefPixels):
        for j in range(undefPixels, m-undefPixels):
            outMag[i][j] = int(round((gradX[i][j]**2 + gradY[i][j]**2)**0.5))
            ang = degrees(atan2(gradY[i][j], gradX[i][j]))
            if ang < 0:    
                outAng[i][j] = ang + 360
            else:
                outAng[i][j] = ang
    return outMag, outAng

# Function for Gaussian Smoothing
def gaussian_smoothing(img, maskSize):
    # Setting mask parameters
    out = np.zeros(img.shape, dtype='uint8')
    if maskSize == 7:
        mask = np.array([[1,1,2,2,2,1,1],
                           [1,2,2,4,2,2,1],
                           [2,2,4,8,4,2,2],
                           [2,4,8,16,8,4,2],
                           [2,2,4,8,4,2,2],
                           [1,2,2,4,2,2,1],
                           [1,1,2,2,2,1,1]], dtype='uint8')
        maskSum = np.uint8(140)
        maskSizeHalf = floor(maskSize/2)
    # Getting img dimensions
    n, m = img.shape
    # Starting gaussian masking
    for i in range(n):
        for j in range(m):
            #i,j=3,223
            iLow = i - maskSizeHalf
            iUp = i + maskSizeHalf + 1
            jLow = j - maskSizeHalf
            jUp = j + maskSizeHalf + 1
            windowPixels = img[iLow:iUp, jLow:jUp]
            # 
            if windowPixels.shape == (maskSize, maskSize):
                #print(i,j)
                out[i,j] = convolution(mask, windowPixels, maskSum)
    return out
            
# Function for Gradient Operation
def gradient_operation(img, maskSize, axis, gaussianSize):
    # Setting mask parameters
    out = np.zeros(img.shape, dtype='int8')
    # Variable to hold undefined area from gaussian filtering
    undefPixels = floor(gaussianSize/2)
    # X axis sobel mask
    if maskSize == 3 and axis.lower() == 'x':
        mask = np.array([[-1,0,1],
                           [-2,0,2],
                           [-1,0,1]], dtype='int8')
        maskSizeHalf = floor(maskSize/2)
    # Y axis sobel mask
    if maskSize == 3 and axis.lower() == 'y':
        mask = np.array([[1,2,1],
                           [0,0,0],
                           [-1,-2,-1]], dtype='int8')
        maskSizeHalf = floor(maskSize/2)
    # Getting img dimensions
    n, m = img.shape
    # Starting gaussian masking
    for i in range(undefPixels+1, n-undefPixels-1):
        for j in range(undefPixels+1, m-undefPixels-1):
            #i,j,maskSizeHalf=4,4,1
            iLow = i - maskSizeHalf
            iUp = i + maskSizeHalf + 1
            jLow = j - maskSizeHalf
            jUp = j + maskSizeHalf + 1
            windowPixels = img[iLow:iUp, jLow:jUp]
            # 
            if windowPixels.shape == (maskSize, maskSize):
                #print(i,j)
                out[i,j] = convolution(mask, windowPixels, 1)
    return out

def non_maxima_suppression():
    pass

def thresholding():
    pass



# MAIN

#path = 'data/Houses-225.bmp'
path = 'data/Zebra-crossing-1.bmp'
img = imread(path, 0) 
resultGauss = gaussian_smoothing(img, 7)
resultGradX = gradient_operation(resultGauss, 3, 'x', 7)
resultGradY = gradient_operation(resultGauss, 3, 'y', 7)
resultGradMag, resultGradAng = gradient_mag_and_angle(resultGradX, resultGradY, 3, 7)

imshow('image', img)
imshow('image', resultGauss)
imshow('image', resultGradX)
imshow('image', resultGradY)
imshow('image', resultGradMag)

waitKey(0)
