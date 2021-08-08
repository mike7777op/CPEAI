import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
import random


def DataSet():
    # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
    # 每个文件夹下又创建了 glue，medicine 两个文件夹，所以这里一共四个路径
  
    
    train_path_1 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/IA/'
    train_path_2 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/IB/'
    train_path_3 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/MDCK/'
    train_path_4 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/Para1/'
    train_path_5 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/Para2/'
    train_path_6 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/Para3/' 
    train_path_7 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/MK2/'
    train_path_8 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/EV/'    
    train_path_9 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/RD/'
    train_path_10 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/RSV/'
    train_path_11 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/Hep-2/'
    train_path_12 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/ADV/'
    train_path_13 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/train/A549/'

    test_path_1 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/IA/'
    test_path_2 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/IB/'
    test_path_3 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/MDCK/'
    test_path_4 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/Para1/'
    test_path_5 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/Para2/'
    test_path_6 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/Para3/'
    test_path_7 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/MK2/'    
    test_path_8 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/EV/'
    test_path_9 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/RD/'
    test_path_10 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/RSV/'
    test_path_11 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/Hep-2/'
    test_path_12 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/ADV/'
    test_path_13 = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/test/A549/'


    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    imglist_train_1 = os.listdir(train_path_1)
    imglist_train_2 = os.listdir(train_path_2)
    imglist_train_3 = os.listdir(train_path_3)
    imglist_train_4 = os.listdir(train_path_4)
    imglist_train_5 = os.listdir(train_path_5)
    imglist_train_6 = os.listdir(train_path_6)
    imglist_train_7 = os.listdir(train_path_7)
    imglist_train_8 = os.listdir(train_path_8)
    imglist_train_9 = os.listdir(train_path_9)
    imglist_train_10 = os.listdir(train_path_10)
    imglist_train_11 = os.listdir(train_path_11)
    imglist_train_12 = os.listdir(train_path_12)
    imglist_train_13 = os.listdir(train_path_13)
    
    # 下面代码读取了 test/ 下的所有图片文件名
    imglist_test_1 = os.listdir(test_path_1)
    imglist_test_2 = os.listdir(test_path_2)
    imglist_test_3 = os.listdir(test_path_3)
    imglist_test_4 = os.listdir(test_path_4)
    imglist_test_5 = os.listdir(test_path_5)
    imglist_test_6 = os.listdir(test_path_6)
    imglist_test_7 = os.listdir(test_path_7)
    imglist_test_8 = os.listdir(test_path_8)
    imglist_test_9 = os.listdir(test_path_9)
    imglist_test_10 = os.listdir(test_path_10)
    imglist_test_11 = os.listdir(test_path_11)
    imglist_test_12 = os.listdir(test_path_12)
    imglist_test_13 = os.listdir(test_path_13)
    
    
    
    # X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) , 224, 224, 3))
    # Y_train = np.empty((len(imglist_train_1) + len(imglist_train_2) , 2))

    #Influ,Para,EV data
    # X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9), 224, 224, 3))
    # Y_train1 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9), 4))
    # Y_train2 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9), 5))
    # Y_train3 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9), 3))
   #Influ,Para,EV,RSV data
    # X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11), 224, 224, 3))
    # Y_train1 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11), 4))
    # Y_train2 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11), 5))
    # Y_train3 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11), 3))
    # Y_train4 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11), 3))
    
    #Influ,Para,EV,RSV,ADV data
    X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 224, 224, 3))
    Y_train1 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 4))
    Y_train2 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 5))
    Y_train3 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 3))
    Y_train4 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 3))
    Y_train5 = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4) + len(imglist_train_5) + len(imglist_train_6) + len(imglist_train_7) + len(imglist_train_8) + len(imglist_train_9) + len(imglist_train_10) + len(imglist_train_11) + len(imglist_train_12) + len(imglist_train_13), 3))
    # count 對像用来計数，每添加一張圖片便加 1
    count = 0
    # 遍历 /train/ 下所有图片
    for img_name in imglist_train_1:
        # 得到图片的路径
        img_path = train_path_1 + img_name
        # 通过 image.load_img() 函数讀取對應的图片，並轉換成目標大小
        #  image 是 tensorflow.keras.preprocessing 中的一個對象
        img = image.load_img(img_path, target_size=(224, 224))
        # 将圖片轉換成 numpy 数组，並除以 255 ，歸一化
        # 轉換之后 img 的 shape 是 （224，224，3）
        img = image.img_to_array(img) / 255.0
        
        # 将處理好的圖片装進定義好的 X_train 對象中
        X_train[count] = img
        # 将對應的標籤裝進 Y_train 對象中
        Y_train1[count] = np.array((1,0,0,0))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))
        count+=1
    # /train/ 下所有圖片
    for img_name in imglist_train_2:

        img_path = train_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,1,0,0))
        Y_train2[count] = np.array((0,0,0,0,1))         
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_train_3:
        
        img_path = train_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,1,0))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))
        count+=1
    
    for img_name in imglist_train_4:

        img_path = train_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((1,0,0,0,0))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_train_5:

        img_path = train_path_5 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,1,0,0,0))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))        
        count+=1
    for img_name in imglist_train_6:

        img_path = train_path_6 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,1,0,0))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))        
        count+=1

    for img_name in imglist_train_7:

        img_path = train_path_7 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,1,0))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))         
        count+=1
    
    for img_name in imglist_train_8:

        img_path = train_path_8 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((1,0,0))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))         
        count+=1

    for img_name in imglist_train_9:

        img_path = train_path_9 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,1,0))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,0,1))        
        count+=1

    for img_name in imglist_train_10:

        img_path = train_path_10 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((1,0,0))
        Y_train5[count] = np.array((0,0,1))        
        count+=1

    for img_name in imglist_train_11:

        img_path = train_path_11 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,1))
        Y_train4[count] = np.array((0,1,0))
        Y_train5[count] = np.array((0,0,1))        
        count+=1

    for img_name in imglist_train_12:

        img_path = train_path_12 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,0))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((1,0,0))        
        count+=1

    for img_name in imglist_train_13:

        img_path = train_path_13 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train1[count] = np.array((0,0,0,1))
        Y_train2[count] = np.array((0,0,0,0,1))
        Y_train3[count] = np.array((0,0,0))
        Y_train4[count] = np.array((0,0,1))
        Y_train5[count] = np.array((0,1,0))        
        count+=1

    # 測试集的数据，與上面的内容完全相同
    # X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) , 224, 224, 3))
    # Y_test = np.empty((len(imglist_test_1) + len(imglist_test_2) , 2))

    # X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9), 224, 224, 3))
    # Y_test1 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9), 4))
    # Y_test2 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9), 5))
    # Y_test3 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9), 3))

    # X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11), 224, 224, 3))
    # Y_test1 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11), 4))
    # Y_test2 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11), 5))
    # Y_test3 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11), 3))
    # Y_test4 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11), 3))
    
    X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 224, 224, 3))
    Y_test1 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 4))
    Y_test2 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 5))
    Y_test3 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 3))
    Y_test4 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 3))
    Y_test5 = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4) + len(imglist_test_5) + len(imglist_test_6) + len(imglist_test_7) + len(imglist_test_8) + len(imglist_test_9) + len(imglist_test_10) + len(imglist_test_11) + len(imglist_test_12) + len(imglist_test_13), 3))
    count = 0
    for img_name in imglist_test_1:

        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((1,0,0,0))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1
        
    for img_name in imglist_test_2:
        
        img_path = test_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,1,0,0))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1
    
    for img_name in imglist_test_3:

        img_path = test_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,1,0))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_4:

        img_path = test_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((1,0,0,0,0))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_5:

        img_path = test_path_5 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,1,0,0,0))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_6:

        img_path = test_path_6 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,1,0,0))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1
    

    for img_name in imglist_test_7:

        img_path = test_path_7 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,1,0))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1
    

    for img_name in imglist_test_8:

        img_path = test_path_8 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((1,0,0))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_9:

        img_path = test_path_9 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,1,0))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_10:

        img_path = test_path_10 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((1,0,0))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_11:

        img_path = test_path_11 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,1,0))
        Y_test5[count] = np.array((0,0,1))
        count+=1

    for img_name in imglist_test_12:

        img_path = test_path_12 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((1,0,0))
        count+=1

    for img_name in imglist_test_13:

        img_path = test_path_13 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test1[count] = np.array((0,0,0,1))
        Y_test2[count] = np.array((0,0,0,0,1))
        Y_test3[count] = np.array((0,0,1))
        Y_test4[count] = np.array((0,0,1))
        Y_test5[count] = np.array((0,1,0))
        count+=1

	# 打亂训练集中的数據
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train1 = Y_train1[index]
    Y_train2 = Y_train2[index]
    Y_train3 = Y_train3[index]
    Y_train4 = Y_train4[index]
    Y_train5 = Y_train5[index]
     
    # 打亂測試集中的數據
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test1 = Y_test1[index]
    Y_test2 = Y_test2[index]
    Y_test3 = Y_test3[index]
    Y_test4 = Y_test4[index]
    Y_test5 = Y_test5[index]
    # y_test = y_test[index]
    	

    # return X_train,Y_train1,Y_train2,Y_train3,Y_train4,X_test,Y_test1,Y_test2,Y_test3,Y_test4
    
    return X_train,Y_train1,Y_train2,Y_train3,Y_train4,Y_train5,X_test,Y_test1,Y_test2,Y_test3,Y_test4,Y_test5


X_train,Y_train1,Y_train2,Y_train3,Y_train4,Y_train5,X_test,Y_test1,Y_test2,Y_test3,Y_test4,Y_test5 = DataSet()
# X_train,Y_train1,Y_train2,Y_train3,Y_train4,X_test,Y_test1,Y_test2,Y_test3,Y_test4 = DataSet()
# X_train,Y_train1,Y_train2,Y_train3,X_test,Y_test1,Y_test2,Y_test3 = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train1 shape : ',Y_train1.shape)
print('Y_train2 shape : ',Y_train2.shape)
print('Y_train3 shape : ',Y_train3.shape)
print('Y_train4 shape : ',Y_train4.shape)
print('Y_train5 shape : ',Y_train5.shape)

print('X_test shape : ',X_test.shape)
print('Y_test1 shape : ',Y_test1.shape)
print('Y_test2 shape : ',Y_test2.shape)
print('Y_test3 shape : ',Y_test3.shape)
print('Y_test4 shape : ',Y_test4.shape)
print('Y_test5 shape : ',Y_test5.shape)

np.save('X_train',X_train)
np.save('Influ_Y_train1',Y_train1)
np.save('Para_Y_train2',Y_train2)
np.save('EV_Y_train3',Y_train3)
np.save('RSV_Y_train4',Y_train4)
np.save('ADV_Y_train5',Y_train5)

# np.save('y_train',y_train)

np.save('X_test',X_test)
np.save('Influ_Y_test1',Y_test1)
np.save('Para_Y_test2',Y_test2)
np.save('EV_Y_test3',Y_test3)
np.save('RSV_Y_test4',Y_test4)
np.save('ADV_Y_test5',Y_test5)
