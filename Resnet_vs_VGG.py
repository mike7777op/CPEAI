import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
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
X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/Y_test.npy')






def model(model= 'VGG16'):
    if model == 'VGG16':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = 9)

        x = base_model.output
        x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    elif model == 'ResNet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = 9
        )
        x = base_model.output
        x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    model_final = Model(inputs=base_model.input, outputs=x)
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True

    model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00003),
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    model_final.summary()
    history = model_final.fit(X_train, Y_train,validation_split=0.25 ,epochs=2, batch_size=32, shuffle=True)
    
    return history

training1 = model('VGG16')
training2 = model('ResNet50')

#train loss,val loss
plt.plot(training1.history['val_loss'],'r-.')
plt.plot(training2.history['val_loss'],'g--^')
plt.title('Single val loss and Multi val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["Single_val_loss","Multi_val_loss"],loc="upper right")
plt.grid(True)
plt.show()


# plt.plot(training1.history['accuracy'],'r-.^')
plt.plot(training1.history['loss'],'r-.')
# plt.plot(training1.history['val_accuracy'],'b--*')
plt.plot(training1.history['val_loss'],'g--^')
plt.title("VGG16 training loss and val loss")
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_loss","val_loss"],loc="upper right")
plt.grid(True)
plt.show()

# plt.plot(training1.history['accuracy'],'r-.^')
plt.plot(training2.history['loss'],'r-.^')
# plt.plot(training1.history['val_accuracy'],'b--*')
plt.plot(training2.history['val_loss'],'g--')
plt.title("ResNet50 training loss and val loss")
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_loss","val_loss"],loc="upper right")
plt.grid(True)
plt.show()

plt.plot(training1.history['loss'],'r-.^')
plt.plot(training1.history['val_loss'],'g--^')
plt.plot(training2.history['loss'],'b--*')
plt.plot(training2.history['val_loss'],'y-o')
plt.title("ResNet50 and VGG16 training loss and val loss")
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["VGG16_training_loss","VGG16_val_loss","ResNet50_training_loss","ResNet50_val_loss"],loc="upper right")
plt.grid(True)
plt.show()