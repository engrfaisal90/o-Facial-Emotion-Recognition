# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 01:24:52 2018

@author: Faisal
"""

import face_extraction as fe
import cv2
img = cv2.imread('test2.JPG',0)
rows,cols = img.shape
 
denoised = cv2.GaussianBlur(img,(5,5),0)
filter = cv2.Laplacian(denoised,cv2.CV_64F)
 
cv2.imshow('Original',img)
cv2.imshow('Laplacian Filter',filter)
 

 
cv2.waitKey(0)