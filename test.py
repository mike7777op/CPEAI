import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
Cell_list = ['RD','MDCK','MK2']
Influ_list = ['IA','IB','MDCK','None']
Para_list = ['Para1','Para2','Para3','MK2','None']
CB1RD_list = ['CB1','RD','None']
ParaIAIBCB1_list = ['CB1','IA','IB','Para1','Para2','Para3']
All_list = ['CB1','IA','IB','Para1','Para2','Para3','MDCK','MK2','RD']

X_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/CB_Y_test3.npy')
#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/IPC_Mutitask_1.h5')

# #test_img
img_path = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test/CB1/773.jpg'

img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维
pred = model.predict(img)[1]
for i in pred:
    top_inds = i.argsort()[::-1][:5]
    print(top_inds)
    for j in top_inds:
        # print(j)
        print('    {:.3f}  {}'.format(i[j], Para_list[j]))
  

