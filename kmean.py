import os
import shutil
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.python.keras.applications import ResNet50
#from keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
import glob
from sklearn.cluster import KMeans
import cv2
#import tensorflow



def extract_vector(path):
    resnet_feature_list = []

    #print( "path:",path )
    for im in img_path:
        im = cv2.imread(im)
        im = cv2.resize(im,(224,224))
        img = preprocess_input(np.expand_dims(im.copy(), axis=0))
        resnet_feature = my_new_model.predict(img)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())

    return np.array(resnet_feature_list)




if __name__ == '__main__':
    #config
    img_type = '.png'
    source_dir = r'D:\hobin\AOI_train_test_data(all)\train\5\*' + img_type # all english not chinese
    outputpath = r'D:\hobin\AOI_train_test_data(kmean_after 100)\train\kmean_result\5'
    cluster_num = 10
    resnet_weights_path = r'C:\Users\Public_B\Desktop\kmean\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    
    print( "ResNet50_intial...")
    #model boot
    my_new_model = Sequential()
    #my_new_model = tensorflow.keras.Sequential()
    my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    
    # Say not to train first layer (ResNet) model. It is already trained
    my_new_model.layers[0].trainable = False
    
    print( "ResNet50_intial...OK")
    
    
    
    
    img_path = glob.glob(source_dir) # input DIR
    
    print( "extract_vector...")
    array = extract_vector( img_path )
    
    print( "extract_vector..OK.")
    print( "KMeans...")
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(array)
    #print(kmeans.labels_)
    
    print( "KMeans...OK")
    list_img_clu =  kmeans.labels_.tolist()
    class_path = outputpath  #output DIR
    mk_dir = set(list_img_clu)  #list to set 
    
    
    # output
    if not os.path.isdir(class_path): #if result class dir does not exist
        os.mkdir(class_path) #then create 
    	
    for i in mk_dir: #create result class dir
        if not os.path.isdir(os.path.join( class_path ,'Class' + str(i)) ): #if result class dir does not exist
            os.mkdir(os.path.join( class_path ,'Class' + str(i)) ) #then create 
        
    for i in range( len(img_path) ): #copy file 
        file,filename = os.path.split( img_path[i] )  #split file, filename
        #shutil.copyfile(img_path[i],os.path.join(class_path,'Class' + str(list_img_clu[i]),'C_' + str(list_img_clu[i])+'_'+ filename) ) #copy file
        shutil.copyfile(img_path[i],os.path.join(class_path,'Class' + str(list_img_clu[i]),filename) ) #copy file without change name


