# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 00:06:13 2018

@author: Faisal
"""

import numpy as np
import dlib
from imutils import face_utils
import cv2
import os

def face_remap(shape):
    remapped_image = cv2.convexHull(shape)
    return remapped_image
def crop_image(img,tol=50):
    # img is image data
    # tol  is tolerance
    mask = img>tol
    img= img[np.ix_(mask.any(1),mask.any(0))]
     
    return img

def intensity_norm(img):
    
    blur = cv2.bilateralFilter(img,9,75,75)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    im = cv2.filter2D(blur, -1, kernel)
    cl1=cv2.equalizeHist(im)
    blur1 = cv2.bilateralFilter(cl1,9,100,100)
    
    
    #res=np.hstack(blur,im,cl1,blur1)
    
    #cv2.imshow("Image", res)
    #cv2.waitKey(0)
    return blur,im,cl1,blur1
    


def preprocessing(image,predictor):
    #image = cv2.imread("test_images/allignedtest.jpg",0)
    #print(image)
    allfaces=[]
    out_face = np.zeros_like(image)
    detector = dlib.get_frontal_face_detector()   
    
    # detect faces in the grayscale image
    rects = detector(image, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
    
       #initialize mask array
        remapped_shape = np.zeros_like(shape) 
        feature_mask = np.zeros((image.shape[0], image.shape[1]))   
    
       # we extract the face
        remapped_shape = face_remap(shape)
        remapped_shape = face_remap(shape)
       #print(remapped_shape)
       
        cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
       
        feature_mask = feature_mask.astype(np.bool)
        out_face[feature_mask] = image[feature_mask]
       #crop the face
        out_face=crop_image(out_face,tol=10)
        allfaces.append(out_face)
        #cv2.imshow("Image", out_face)
        #cv2.waitKey(0)
        
        
    return allfaces
        

