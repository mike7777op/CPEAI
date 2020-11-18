import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
Para_list = ['Para1','Para2','Para3','MK2']
Influ_list = ['IA','IB','MDCK']
IAIBCB1_list = ['CB1','IA','IB']

# model = ResNet101(
#     weights=None,
#     classes=4
# )
# 加载训练好的模型
# model.load_weights('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Para1_Mk2_Generator_resnet101_model_2.h5')
#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/IAIBCB1_imagenet_resnet101_model_1.h5')

img_path = '/home/pmcn/workspace/Test_Code/Resnet50/IAIBCB1_data/test/IB/320.jpg'

img = image.load_img(img_path, target_size=(224, 224))

# plt.imshow(img)
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维
pred = model.predict(img)[0]
top_inds = pred.argsort()[::-1][:5]
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], IAIBCB1_list[i]))
