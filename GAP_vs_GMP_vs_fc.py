import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Dense, Dropout,add
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
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
from keras.optimizers import Adam


K.clear_session()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)




X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/test_npy/Y_test.npy')


print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)

def model(model = 'avg'):
    if model == 'avg':
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
        

    elif model == 'max':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='max',
            classes = 9)

        x = base_model.output
        x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    elif model == 'fc':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = 9)

        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)
        

    model_final = Model(inputs=base_model.input, outputs=x)
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True
    learning_rate = 0.00001 
    optimizer = Adam(lr=learning_rate)
    model_final.compile(#optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                optimizer=optimizer,
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    model_final.summary()
    history = model_final.fit(X_train, Y_train,validation_split=0.25 ,epochs=100, batch_size=32, shuffle=True)
    preds = model_final.evaluate(X_test, Y_test)
    acc = preds[1]
    loss = preds[0]
    
    return history,acc,loss

avg_history,avg_acc = model('avg')
max_history,max_acc = model('max')
fc_history,fc_acc = model('fc')

plt.plot(avg_history.history['val_accuracy'])
plt.plot(max_history.history['val_accuracy'])
plt.plot(fc_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch \nAVG accuracy : {:0.4f}; MAX accuracy : {:0.4f}; fc accuracy : {:0.4f}'.format(avg_acc,max_acc,fc_acc))
plt.legend(['AVG','MAX','FC'],loc="lower right")
plt.grid(True)
plt.show()

plt.plot(avg_history.history['val_loss'])
plt.plot(max_history.history['val_loss'])
plt.plot(fc_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['AVG','MAX','FC'],loc="upper right")
plt.grid(True)
plt.show()