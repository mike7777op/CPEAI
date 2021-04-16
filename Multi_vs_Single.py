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





def model(model= 'Single'):
    if model == 'Single':
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
        learning_rate = 0.00003 
        optimizer = Adam(learning_rate=learning_rate)
        model_final.compile(optimizer=optimizer,
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        model_final.summary()

        training1 = model_final.fit(X_train_s, Y_train_s ,validation_split=0.25,epochs=2, batch_size=32, shuffle=True)

        return training1


    elif model == 'Multi':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = [4,5,3]
        )
        x1 = base_model.output
        x1 = Dropout(0.5)(x1)
        Influ = Dense(4,activation='softmax', name='softmax1')(x1)
        Para = Dense(5, activation='softmax', name='softmax2')(x1)
        EV = Dense(3, activation='softmax', name='softmax3')(x1)

        model_final = Model(inputs=base_model.input,outputs=[Influ,Para,EV])
        # opt = keras.optimizers.Adam(learning_rate=0.00001)
        learning_rate = 0.00003 
        optimizer = Adam(lr=learning_rate)
        model_final.compile(optimizer=optimizer,
                    # optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                    # loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy, 'softmax4':categorical_crossentropy},
                    loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy},
                    metrics=['accuracy'])
        model_final.summary()

        training2 = model_final.fit(X_train,[Y_train1,Y_train2,Y_train3],validation_split=0.25,epochs=2, batch_size=32, shuffle=True)

        return training2


training1 = model('Single')
training2 = model('Multi')

#train loss,val loss
plt.plot(training1.history['val_loss'],'r-.^')
plt.plot(training2.history['val_loss'],'g--')
plt.title('Single val loss and Multi val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["Single_val_loss","Multi_val_loss"],loc="upper left")
plt.grid(True)
plt.show()


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

