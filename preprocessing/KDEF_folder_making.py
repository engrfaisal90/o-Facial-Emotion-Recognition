# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 21:53:23 2018

@author: Faisal
"""

import cv2
import os

os.chdir('..\..')
curpath=os.getcwd()
readpath=curpath+'\\KDEF'
outputpath=curpath+'\\KDEF classified\\'
print(readpath)
print(outputpath)
inputpath='D:/Educational Data/tigp phd data/Projects/MM Project/KDEF/'
outputpath='D:/Educational Data/tigp phd data/Projects/MM Project/KDEF classified/'
for root, dirs, files in os.walk(readpath):
        path_list = root.split(os.sep)
        print(root)
        for name in files:
            print(root+"/"+name)
#            filename, file_extension = os.path.splitext(name)
#            if (file_extension==".JPG" or file_extension==".jpg"):
#                img = cv2.imread(root+"/"+name)
#                gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                if(name[4]+name[5]=='AF'):
#                    cv2.imwrite(outputpath+"/2/"+name, gray_image)
#                if(name[4]+name[5]=='AN'):
#                    cv2.imwrite(outputpath+"/0/"+name, gray_image)
#                if(name[4]+name[5]=='DI'):
#                    cv2.imwrite(outputpath+"/1/"+name, gray_image)
#                if(name[4]+name[5]=='SU'):
#                    cv2.imwrite(outputpath+"/5/"+name, gray_image)
#                if(name[4]+name[5]=='HA'):
#                    cv2.imwrite(outputpath+"/3/"+name, gray_image)
#                if(name[4]+name[5]=='NE'):
#                    cv2.imwrite(outputpath+"/6/"+name, gray_image)
#                if(name[4]+name[5]=='SA'):
#                    cv2.imwrite(outputpath+"/4/"+name, gray_image)