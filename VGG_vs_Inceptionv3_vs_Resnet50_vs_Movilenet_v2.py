import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Flatten, Dense, Dropout, concatenate
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
from keras import backend as K

K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
K.set_session(session)


X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/Y_test.npy')


print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)

def model(model= 'VGG'):
    if model == 'VGG':
        base_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = 9)

        x = base_model.output
        x = Flatten()(x)
        # x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    elif model == 'Inception_v3':
        base_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = 9
        )
        x = base_model.output
        x = Flatten()(x)
        # x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    elif model == 'ResNet50':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = 9)

        x = base_model.output
        x = Flatten()(x)
        # x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    elif model == 'MoblileNetv2':
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = 9)

        x = base_model.output
        x = Flatten()(x)
        # x = Dropout(0.5)(x)
        x = Dense(9, activation='softmax', name='softmax')(x)

    model_final = Model(inputs=base_model.input, outputs=x)
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True

    model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    model_final.summary()
    history = model_final.fit(X_train, Y_train,validation_split=0.25 ,epochs=100, batch_size=32, shuffle=True)
    preds = model_final.evaluate(X_test, Y_test)
    acc = preds[1]
    loss = preds[0]
    
    return history,acc,loss

Vgg_history,Vgg_acc,Vgg_loss = model('VGG')
Inceptionv3_history,Inceptionv3_acc,Inceptionv3_loss = model('Inception_v3')
ResNet50_history,Resnet50_acc,Resnet50_loss = model('ResNet50')
MobileNetV2_history,MobileNetV2_acc,MobileNetV2_loss = model('MoblileNetv2')

plt.plot(Vgg_history.history['val_accuracy'],'r-.^')
plt.plot(Inceptionv3_history.history['val_accuracy'],'g--D')
plt.plot(ResNet50_history.history['val_accuracy'],'b--*')
plt.plot(MobileNetV2_history.history['val_accuracy'],'y-o')
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch \nVGG16 accuracy : {:0.4f}; InceptionV3 accuracy : {:0.4f}; ResNet50 accuracy : {:0.4f}; MobileNetV2 accuracy : {:0.4f}'.format(Vgg_acc,Inceptionv3_acc,Resnet50_acc,MobileNetV2_acc))
plt.legend(['VGG16','InveptionV3','ResNet50','MobileNetV2'],loc="lower right")
plt.grid(True)
plt.show()

plt.plot(Vgg_history.history['val_loss'],'r-.^')
plt.plot(Inceptionv3_history.history['val_loss'],'g--D')
plt.plot(ResNet50_history.history['val_loss'],'b--*')
plt.plot(MobileNetV2_history.history['val_loss'],'y-.')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['VGG16','Inceptin_V3','ResNet50','MobileNetV2'],loc="upper right")
plt.grid(True)
plt.show()

print(Vgg_history.history['loss'][99])
print(Vgg_history.history['val_loss'][99])

print(Inceptionv3_history.history['loss'][99])
print(Inceptionv3_history.history['val_loss'][99])

print(ResNet50_history.history['loss'][99])
print(ResNet50_history.history['val_loss'][99])

print(MobileNetV2_history.history['loss'][99])
print(MobileNetV2_history.history['val_loss'][99])

