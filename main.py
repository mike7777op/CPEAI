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
Cell_list = ['MDCK','MK2','RD']
Para_list = ['Para1','Para2','Para3','MK2']
Influ_list = ['IA','IB','MDCK']
CB1RD_list = ['CB1','RD']
ParaIAIBCB1_list = ['CB1','IA','IB','Para1','Para2','Para3']
All_list = ['CB1','IA','IB','Para1','Para2','Para3','MDCK','MK2','RD']
MDCK = 'MDCK'
MK2 = 'MK2'
RD = 'RD'
print(type(MDCK))
#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/CV/Cell/Cell_1.h5')

#test_img
img_path = '/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train/IB/8.jpg'

img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维
pred = model.predict(img)[0]
top_inds = pred.argsort()[::-1][:5]
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], Cell_list[i]))
index = top_inds[0]
Cell = Cell_list[index]
if Cell == MDCK:
    model1 = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/CV/MDCK/IAIBMDCK_1.h5')
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维
    pred = model1.predict(img)[0]
    top_inds1 = pred.argsort()[::-1][:5]
    for i in top_inds1:
        print('    {:.3f}  {}'.format(pred[i], Influ_list[i]))

elif Cell == MK2:
    model2 = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/CV/MK2/ParaMK2_2.h5')
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维
    pred = model2.predict(img)[0]
    top_inds2 = pred.argsort()[::-1][:5]
    for i in top_inds2:
        print('    {:.3f}  {}'.format(pred[i], Para_list[i]))
elif Cell == RD:
    model3 = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/CV/RD/CB1RD_1.h5')
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # 为batch添加第四维
    pred = model3.predict(img)[0]
    top_inds3 = pred.argsort()[::-1][:5]
    for i in top_inds3:
        print('    {:.3f}  {}'.format(pred[i], CB1RD_list[i]))
# for i in top_inds:
    # print('    {:.3f}  {}'.format(pred[i], Cell_list[i]))
