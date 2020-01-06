# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 01:08:36 2018

@author: Faisal
"""

import image_allignment as ia
import face as pre
import cv2
import os
import dlib
inputpath="\stra_images"
predictor_path="shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
os.chdir("..")
cwd = os.getcwd()
readpath=cwd+inputpath

for root, dirs, files in os.walk(readpath):
        path_list = root.split(os.sep)
        outputpath=str(cwd+"/pre images/"+path_list[-2]+"/"+path_list[-1])
        if(len(files)>0):
            try:
                os.makedirs(outputpath)
            except:
                print("Already created")
        for name in files:
            img = cv2.imread(root+"\\"+name)
            #cv2.imshow("Image", img)
            #cv2.waitKey(0)
            print(root)
            
            #rotatedimage=ia.allignment(img)
            try:
                im1,im2,im3,im4=pre.preprocessing(img,predictor)
                if(im1 is not None):
                    #cv2.imshow("Image", preprocessed)
                    #cv2.waitKey(0)
                    print(outputpath+"/"+name)
                    cv2.imwrite(outputpath+"/1_"+name, im1)
                    cv2.imwrite(outputpath+"/2_"+name, im2)
                    cv2.imwrite(outputpath+"/3_"+name, im3)
                    cv2.imwrite(outputpath+"/4_"+name, im4)
            except:
                continue
            
            #filename, file_extension = os.path.splitext(name)
            #print(root+"/"+filename+"ro.jpg")
            #cv2.imwrite(root+"/"+filename+"ro.jpg", rotatedimage)