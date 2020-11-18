import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)



def DataSet():
    # 首先需要定义训练集和测试集的路径，这里创建了 train ， 和 test 文件夹
    # 每个文件夹下又创建了 glue，medicine 两个文件夹，所以这里一共四个路径
    train_path_1 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/CB1/'
    train_path_2 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/IA/'
    train_path_3 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/IB/'
    train_path_4 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/Para1/'
    train_path_5 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/Para2/'
    train_path_6 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/Para3/'
    train_path_7 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/MDCK/'
    train_path_8 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/MK2/'
    train_path_9 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/RD/'
    
    # test_path_1 = '/home/pmcn/workspace/Test_Code/Resnet50/Generator_data/test/MK2/'
    # test_path_2 = '/home/pmcn/workspace/Test_Code/Resnet50/Generator_data/test/Para1/'
    test_path_1 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/CB1/'
    test_path_2 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/IA/'
    test_path_3 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/IB/'
    test_path_4 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/Para1/'
    test_path_5 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/Para2/'
    test_path_6 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/Para3/'
    test_path_7 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/MDCK/'
    test_path_8 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/MK2/'
    test_path_9 = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/RD/'
    
    # os.listdir(path) 是 python 中的函数，它会列出 path 下的所有文件名
    imglist_train_1 = os.listdir(train_path_1)
    imglist_train_2 = os.listdir(train_path_2)
    imglist_train_3 = os.listdir(train_path_3)
    imglist_train_4 = os.listdir(train_path_4)
    
    # 下面代码读取了 test/ 下的所有图片文件名
    imglist_test_1 = os.listdir(test_path_1)
    imglist_test_2 = os.listdir(test_path_2)
    imglist_test_3 = os.listdir(test_path_3)
    imglist_test_4 = os.listdir(test_path_4)
    
    # 这里定义两个 numpy 对象，X_train 和 Y_train
    
    # X_train 对象用来存放训练集的图片。每张图片都需要转换成 numpy 向量形式
    # X_train 的 shape 是 (360，224，224，3) 
    # 360 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为 resnet 要求输入的图片尺寸是 (224,224) , 所以要设置成相同大小（也可以设置成其它大小，参看 keras 的文档）
    # 3 是图片的通道数（rgb）
    
    # Y_train 用来存放训练集中每张图片对应的标签
    # Y_train 的 shape 是 （360，2）
    # 360 是训练集中图片的数量（训练集中固体胶和西瓜霜图片数量之和）
    # 因为一共有两种图片，所以第二个维度设置为 2
    # Y_train 大概是这样的数据 [[0,1],[0,1],[1,0],[0,1],...]
    # [0,1] 就是一张图片的标签，这里设置 [1,0] 代表 固体胶，[0,1] 代表西瓜霜
    # 如果你有三类图片 Y_train 就因该设置为 (your_train_size,3)
    
    X_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_1) + len(imglist_train_2) + len(imglist_train_3) + len(imglist_train_4), 4))
    
    # count 对象用来计数，每添加一张图片便加 1
    count = 0
    # 遍历 /train/ 下所有图片
    for img_name in imglist_train_1:
        # 得到图片的路径
        img_path = train_path_1 + img_name
        # 通过 image.load_img() 函数读取对应的图片，并转换成目标大小
        #  image 是 tensorflow.keras.preprocessing 中的一个对象
        img = image.load_img(img_path, target_size=(224, 224))
        # 将图片转换成 numpy 数组，并除以 255 ，归一化
        # 转换之后 img 的 shape 是 （224，224，3）
        img = image.img_to_array(img) / 255.0
        
        # 将处理好的图片装进定义好的 X_train 对象中
        X_train[count] = img
        # 将对应的标签装进 Y_train 对象中，这里都是 固体胶（glue）图片，所以标签设为 [1,0]
        Y_train[count] = np.array((1,0,0,0))
        count+=1
    # 遍历 /train/ 下所有图片
    for img_name in imglist_train_2:

        img_path = train_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,1,0,0))
        count+=1

    for img_name in imglist_train_3:
        
        img_path = train_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,0,1,0))
        count+=1
    
    for img_name in imglist_train_4:

        img_path = train_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_train[count] = img
        Y_train[count] = np.array((0,0,1,0))
        count+=1
    # 下面的代码是准备测试集的数据，与上面的内容完全相同，这里不再赘述
    X_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_1) + len(imglist_test_2) + len(imglist_test_3) + len(imglist_test_4), 4))
    count = 0
    for img_name in imglist_test_1:

        img_path = test_path_1 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((1,0,0,0))
        count+=1
        
    for img_name in imglist_test_2:
        
        img_path = test_path_2 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,1,0,0))
        count+=1
    
    for img_name in imglist_test_3:

        img_path = test_path_3 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,0,1,0))
        count+=1

    for img_name in imglist_test_4:

        img_path = test_path_4 + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        
        X_test[count] = img
        Y_test[count] = np.array((0,0,0,1))
        count+=1
	# 打乱训练集中的数据
    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]
    
    # 打乱测试集中的数据
    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]    
    Y_test = Y_test[index]	

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = DataSet()
print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)

model = ResNet50(
    include_top=True,
    weights=None,
    input_tensor=None, 
    input_shape=None,
    pooling=None,
    classes = 4
)


model.compile(optimizer=tf.train.AdamOptimizer(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
traning = model.fit(X_train, Y_train, epochs=100, batch_size=64)
model.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/ParaSeries_resnet_model_1.h5')
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

plt.plot(traning.history['accuracy'])
plt.plot(traning.history['loss'])
plt.title('models accuracy and loss')
plt.show()