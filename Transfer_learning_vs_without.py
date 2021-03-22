import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
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

def model(model = 'imagenet'):
    if model == 'imagenet':
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

    elif model == 'None':
        base_model = VGG16(
            include_top=False,
            weights=None,
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = 9)

        x = base_model.output
        x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    model_final = Model(inputs=base_model.input, outputs=x)
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True

    learning_rate = 0.00001 
    optimizer = Adam(lr=learning_rate)
    model_final.compile(optimizer=optimizer,
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    model_final.summary()
    history = model_final.fit(X_train, Y_train,validation_split=0.25 ,epochs=100, batch_size=32, shuffle=True)
    preds = model_final.evaluate(X_test, Y_test)
    acc = preds[1]
    loss = preds[0]
    return history,acc,loss

TF_history,TF_acc,TF_loss = model('imagenet')
Without_history,Without_acc,Without_loss = model('None')

plt.plot(TF_history.history['val_accuracy'])
plt.plot(Without_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch \nTF accuracy : {:0.4f}; Without TF accuracy : {:0.4f}'.format(TF_acc,Without_acc))
plt.legend(['Tf','Without_Tf'],loc="lower right")
plt.grid(True)
plt.show()

plt.plot(TF_history.history['val_loss'])
plt.plot(Without_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch \nTF Loss : {:0.5f}; Without TF Loss : {:0.5f}'.format(TF_loss,Without_loss))
plt.legend(['Tf','Without_Tf'],loc="upper right")
plt.grid(True)
plt.show()