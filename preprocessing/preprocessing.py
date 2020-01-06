# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:09:01 2018

@author: Faisal
"""

import numpy as np
import dlib
from imutils import face_utils
import cv2
import os
import face_extraction as fe

def outpath(root,i):
        try:
            path_list = root.split("\\")
            if(i%10==0):
                outputpath=str(curpath+"\\traintest dataset\\test\\"+path_list[-1])
            else:
                outputpath=str(curpath+"\\traintest dataset\\Train\\"+path_list[-1])

            os.makedirs(outputpath)
        except:
            print("Already created")
        return outputpath
    
def images_write(im1,im2,im3,im4):
    cv2.imwrite(outputpath+"/1_"+name, im1)
    cv2.imwrite(outputpath+"/2_"+name, im2)
    cv2.imwrite(outputpath+"/3_"+name, im3)
    cv2.imwrite(outputpath+"/4_"+name, im4)



os.chdir('..\..')

curpath=os.getcwd()
readpath=curpath+'\whole dataset'
print(readpath)
predictor_path="code\preprocessing\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

for root, dirs, files in os.walk(readpath):
        i=0
        if(len(files)>0):             
            for name in files:    
                i=i+1
                outputpath=outpath(root,i) 
                img = cv2.imread(root+"\\"+name,0)
                all_faces=fe.preprocessing(img,predictor)
                print(len(all_faces))
                for each_image in all_faces: 
                    if each_image is not None:
                        im1,im2,im3,im4=fe.intensity_norm(each_image)
                        #images_write(im1,im2,im3,im4)
                        res=np.hstack((im1,im2,im3,im4))
                        cv2.imshow("Image", res)
                        cv2.waitKey(0)
#                        print(i)
#                        print(outputpath+"/"+name)
                        
                
        