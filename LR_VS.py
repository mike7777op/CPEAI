import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, Dropout,add
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import ResNet101
from keras.losses import categorical_crossentropy
from PIL import Image
import keras
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import itertools
import visualkeras
from collections import defaultdict
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,SGD


K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# tf.keras.backend.set_session(sess)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
K.set_session(session)



#single
X_train_s = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/X_train.npy')
Y_train_s = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/Y_train.npy')
X_test_s = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/X_test.npy')
Y_test_s = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/Y_test.npy')

#Multi
X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/train_npy/X_train.npy')
Y_train1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/train_npy/Influ_Y_train1.npy')
Y_train2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/train_npy/Para_Y_train2.npy')
Y_train3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/train_npy/EV_Y_train3.npy')

X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/test_npy/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/test_npy/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/test_npy/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Multi/test_npy/EV_Y_test3.npy')





def model(lr= '0.0001'):
    base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = 9)

    x = base_model.output
    x = Dropout(0.5)(x)
    x = Dense(9, activation='softmax', name='softmax')(x)


    model_final = Model(inputs=base_model.input, outputs=x)

    if lr == '0.0001':
        learning_rate = 0.0001 

    elif lr == '0.00001':
        learning_rate= 0.00001

    elif lr == '0.00005':
        learning_rate= 0.00005

    elif lr == '0.00003':
        learning_rate= 0.00003

    elif lr == '0.00002':
        learning_rate= 0.00002


    optimizer = Adam(learning_rate=learning_rate)
    model_final.compile(optimizer=optimizer,
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    model_final.summary()

    training = model_final.fit(X_train_s, Y_train_s ,validation_split=0.25,epochs=100, batch_size=32, shuffle=True)

    return training




training1 = model('0.0001')
training2 = model('0.00001')
training3 = model('0.00005')
training4 = model('0.00003')
training5 = model('0.00002')


#train loss,val loss
plt.plot(training1.history['val_loss'],'r-.^')
plt.plot(training2.history['val_loss'],'g--D')
plt.plot(training3.history['val_loss'],'b--*')
plt.plot(training4.history['val_loss'],'y-o')
plt.plot(training5.history['val_loss'],'c--s')
plt.title('Single val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["Lr:0.0001","Lr:0.00001","Lr:0.00005","Lr:0.00003","Lr:0.00002"],loc="upper right")
plt.grid(True)
plt.show()

plt.plot(training1.history['val_accuracy'],'r-.^')
plt.plot(training2.history['val_accuracy'],'g--D')
plt.plot(training3.history['val_accuracy'],'b--*')
plt.plot(training4.history['val_accuracy'],'y-o')
plt.plot(training5.history['val_accuracy'],'c--s')
plt.title('Single val accuracy')
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["Lr:0.0001","Lr:0.00001","Lr:0.00005","Lr:0.00003","Lr:0.00002"],loc="upper left")
plt.grid(True)
plt.show()

print(training1.history['loss'][99])
print(training1.history['val_loss'][99])

print(training2.history['loss'][99])
print(training2.history['val_loss'][99])

print(training3.history['loss'][99])
print(training3.history['val_loss'][99])

print(training4.history['loss'][99])
print(training4.history['val_loss'][99])

print(training5.history['loss'][99])
print(training5.history['val_loss'][99])
# plt.plot(training.history['accuracy'],'r-.^')
# plt.plot(training.history['loss'],'g--')
# plt.plot(training.history['val_accuracy'],'b--*')
# plt.plot(training.history['val_loss'],'y-o')
# plt.title("Resnet50_model_1")
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

