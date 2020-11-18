import os,sys,io
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
from keras.losses import categorical_crossentropy
from PIL import Image
import keras
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import itertools

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

n = 'MK2'
classes = 3

def train_mutitask(name,classes):
    if name=='MDCK':
        model1 = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = classes
        )

        x1 = model1.output
        x1 = Dropout(0.5)(x1)

        return x1,model1.input
    elif name == 'MK2':
        model2 = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = classes
        )

        x2 = model2.output
        x2 = Dropout(0.5)(x2)

        return x2,model2.input
    elif name == 'RD':
        model3 = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = classes
        )

        x3 = model3.output
        x3 = Dropout(0.5)(x3)

        return x3,model3.input
def final_model(n,classes):
    if n == 'MDCK':

        x,y= train_mutitask(name='MDCK',classes=classes)
        x = Dense(classes, activation='softmax', name='softmax')(x)

        model_final = Model(inputs=y, outputs=x)
        # # for layer in model_final.layers[:FREEZE_LAYERS]:
        # #     layer.trainable = False
        # # for layer in model_final.layers[FREEZE_LAYERS:]:
        # #     layer.trainable = True

        model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                loss=categorical_crossentropy,
                metrics=['accuracy'])

        model_final.summary()
    
    elif n == 'MK2':

        x,y= train_mutitask(name='MK2',classes=classes)
        x = Dense(classes, activation='softmax', name='softmax')(x)

        model_final = Model(inputs=y, outputs=x)
        # # for layer in model_final.layers[:FREEZE_LAYERS]:
        # #     layer.trainable = False
        # # for layer in model_final.layers[FREEZE_LAYERS:]:
        # #     layer.trainable = True

        model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                loss=categorical_crossentropy,
                metrics=['accuracy'])

        model_final.summary()

    elif n == 'RD':

        x,y= train_mutitask(name='RD',classes=classes)
        x = Dense(classes, activation='softmax', name='softmax')(x)

        model_final = Model(inputs=y, outputs=x)
        # # for layer in model_final.layers[:FREEZE_LAYERS]:
        # #     layer.trainable = False
        # # for layer in model_final.layers[FREEZE_LAYERS:]:
        # #     layer.trainable = True

        model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                loss=categorical_crossentropy,
                metrics=['accuracy'])

        model_final.summary()

if __name__ == '__main__':
    final = final_model(n,classes)

    

    # traning = model_final.fit(X_train, Y_train, validation_split=0.2 ,epochs=100, batch_size=32, shuffle=True)
    # model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/All_AVG_bt32_1.h5')
    # preds = model_final.evaluate(X_test, Y_test)
    # print ("Loss = " + str(preds[0]))
    # print ("Test Accuracy = " + str(preds[1]))

    # plt.plot(traning.history['accuracy'])
    # plt.plot(traning.history['loss'])
    # plt.title('models accuracy and loss')
    # plt.show()
