# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 00:03:41 2018

@author: Faisal
"""

import os
import cv2
import numpy as np
import pickle

os.chdir("..\..")
curpath = os.getcwd()
readpath=curpath+"\\last_traintest"
outputpath=curpath+'\\pickledata\\'
Train_data = []
Train_label= []
private_test_data=[]
private_test_label=[]
public_test_data=[]
public_test_label=[]

i=0
for root, dirs, files in os.walk(readpath):
    path_list = root.split(os.sep)
    print(root)
    if(path_list[-2])=='train':
        for name in files:
            if(len(files)>0):
                img = cv2.imread(root+"\\"+name, cv2.IMREAD_GRAYSCALE)
                
                img = cv2.resize(img, (80, 100))
                print(img.shape)

                Train_data.append (img)
                Train_label.append(int(path_list[-1]))
                i=i+1
                #print(i)
    if(path_list[-2])=='test':
        for name in files:
            if(len(files)>0):
                img = cv2.imread(root+"\\"+name, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (80, 100))
                print(img.shape)
                public_test_data.append (img)
                public_test_label.append(int(path_list[-1]))
                i=i+1
                #print(i)
#    if(path_list[-2])=='PrivateTest':
#        for name in files:
#            if(len(files)>0):
#                img = cv2.imread(root+"\\"+name, cv2.IMREAD_GRAYSCALE)
#                private_test_data.append (img)
#                private_test_label.append(int(path_list[-1]))
#                i=i+1
                #print(i)
                

print('Train_data shape:', np.array(Train_data).shape)
print('Train_label shape:', np.array(Train_label).shape)
print('public test data shape:', np.array(public_test_data).shape)
print('public test label shape:', np.array(public_test_label).shape)
#print('private test data shape:', np.array(private_test_data).shape)
#print('private test label shape:', np.array(private_test_label).shape)
#with open(outputpath+"nd_train_data", "wb") as f:
#	pickle.dump(Train_data , f)
#with open(outputpath+"nd_train_labels", "wb") as f:
#	pickle.dump(Train_label, f)
#with open(outputpath+"nd_test_data", "wb") as f:
#	pickle.dump(public_test_data, f)
#with open(outputpath+"nd_test_label", "wb") as f:
#	pickle.dump(public_test_label, f)
#with open("o_private_test_data", "wb") as f:
#	pickle.dump(private_test_data, f)
#with open("o_private_test_label", "wb") as f:
#	pickle.dump(private_test_label, f)

    