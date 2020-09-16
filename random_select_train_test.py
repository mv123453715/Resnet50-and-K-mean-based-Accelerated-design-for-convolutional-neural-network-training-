# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:50:28 2020

@author: User
"""

import os
import shutil
#from keras.applications.vgg16 import preprocess_input
import numpy as np
#from tensorflow.python.keras.applications import ResNet50
#from keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.models import Sequential
import glob
from sklearn.cluster import KMeans
import cv2
#import tensorflow
from random import sample

if __name__ == '__main__':
    img_type = '.png'
    source_dir = r'D:\hobin\AOI_train_test_data(kmean_after 100)\train\kmean_result\5'
    subdir = os.listdir(source_dir)
    random_num = 2
    print(subdir)
    
    for i in subdir:
        imgdir = os.path.join(source_dir,i)
        imgdir = imgdir +'\*' + img_type
        img_path = glob.glob(imgdir)
        #print(img_path)
        if len(img_path) < random_num:
            random_num = len(img_path) 
        samplelist = sample(img_path, random_num)
        print(samplelist)
        
        des_train_dir = os.path.join(source_dir,"train")
        if ( os.path.exists(des_train_dir) == False  ):
            os.mkdir(des_train_dir)
        for j in samplelist:
            shutil.move(j,des_train_dir)
        
        des_train_dir_img_path = glob.glob(des_train_dir+'\*' + img_type)

"""        
        ran_num = 5
        if len(des_train_dir_img_path) < ran_num:
            ran_num = len(des_train_dir_img_path) 
            
        des_train_dir_samplelist = sample(des_train_dir_img_path, ran_num)
        des_test_dir = os.path.join(source_dir,"test")
        if ( os.path.exists(des_test_dir) == False  ):
            os.mkdir(des_test_dir)
        for j in des_train_dir_samplelist:
            shutil.move(j,des_test_dir)
            
            
        imgdir = os.path.join(source_dir,i)
        imgdir = imgdir +'\*' + img_type
        img_path = glob.glob(imgdir)
        
        des_other_dir = os.path.join(source_dir,"other")
        if ( os.path.exists(des_other_dir) == False  ):
            os.mkdir(des_other_dir)
        for j in img_path:
            shutil.move(j,des_other_dir)
"""        
        
        
        
        
    