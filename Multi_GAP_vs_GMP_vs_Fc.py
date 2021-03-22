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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from Fusion import fusion


K.clear_session()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)




X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_train_npy/X_train.npy')
Y_train1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_train_npy/Influ_Y_train1.npy')
Y_train2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_train_npy/Para_Y_train2.npy')
Y_train3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_train_npy/CB_Y_train3.npy')


X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_test_npy/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_test_npy/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_test_npy/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/IPC_test_npy/CB_Y_test3.npy')


print('X_train shape : ',X_train.shape)
print('Y_train1 shape : ',Y_train1.shape)
print('Y_train2 shape : ',Y_train2.shape)
print('Y_train3 shape : ',Y_train3.shape)


print('X_test shape : ',X_test.shape)
print('Y_test1 shape : ',Y_test1.shape)
print('Y_test2 shape : ',Y_test2.shape)
print('Y_test3 shape : ',Y_test3.shape)

def model(model = 'avg'):
    if model == 'avg':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='avg',
            classes = [4,5,3]
        )

        x = base_model.output
        x = Dropout(0.5)(x)
        Influ = Dense(4,activation='softmax', name='softmax1')(x)
        Para = Dense(5, activation='softmax', name='softmax2')(x)
        EV = Dense(3, activation='softmax', name='softmax3')(x)
        

    elif model == 'max':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling='max',
            classes = [4,5,3]
        )

        x = base_model.output
        x = Dropout(0.5)(x)
        Influ = Dense(4,activation='softmax', name='softmax1')(x)
        Para = Dense(5, activation='softmax', name='softmax2')(x)
        EV = Dense(3, activation='softmax', name='softmax3')(x)

    elif model == 'fc':
        base_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None, 
            input_shape=(224,224,3),
            pooling=None,
            classes = [4,5,3]
        )

        x = base_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        Influ = Dense(4,activation='softmax', name='softmax1')(x)
        Para = Dense(5, activation='softmax', name='softmax2')(x)
        EV = Dense(3, activation='softmax', name='softmax3')(x)
        

    model_final = Model(inputs=base_model.input,outputs=[Influ,Para,EV])
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True

    learning_rate = 0.00001 
    optimizer = Adam(lr=learning_rate, decay=0.0)
    model_final.compile(optimizer=optimizer,
                # loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy, 'softmax4':categorical_crossentropy},
                loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy},
                metrics=['accuracy'])
    model_final.summary()
    history = model_final.fit(X_train,[Y_train1,Y_train2,Y_train3],validation_split=0.25,epochs=1, batch_size=32, shuffle=True)
    loss, softmax1_loss, softmax2_loss, softmax3_loss, softmax1_acc, softmax2_acc, softmax3_acc = model_final.evaluate(X_test, [Y_test1,Y_test2,Y_test3])
    fusion(model=model_final)
    
    return history,fusion,softmax1_acc,softmax2_acc,softmax3_acc

avg_history,avg_fusion_acc,avg_1,avg_2,avg_3 = model('avg')
# max_history,max_fusion_acc,max_1,max_2,max_3 = model('max')
# fc_history,fc_fusion_acc,fc_1,fc_2,fc_3 = model('fc')

#Influ 
# plt.plot(avg_history.history['val_softmax1_accuracy'])
# plt.plot(max_history.history['val_softmax1_accuracy'])
# plt.plot(fc_history.history['val_softmax1_accuracy'])
# plt.title('Softmax1 accuracy')
# plt.ylabel('Validation Accuracy')
# plt.xlabel('Epoch \nAVG accuracy : {:0.4f}; MAX accuracy : {:0.4f}; fc accuracy : {:0.4f}'.format(avg_1,max_1,fc_1))
# plt.legend(['AVG','MAX','FC'],loc="lower right")
# plt.grid(True)
# plt.show()

# plt.plot(avg_history.history['val_softmax1_loss'])
# plt.plot(max_history.history['val_softmax1_loss'])
# plt.plot(fc_history.history['val_softmax1_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['AVG','MAX','FC'],loc="upper right")
# plt.grid(True)
# plt.show()

# #Para
# plt.plot(avg_history.history['val_softmax2_accuracy'])
# plt.plot(max_history.history['val_softmax2_accuracy'])
# plt.plot(fc_history.history['val_softmax2_accuracy'])
# plt.title('Softmax2 accuracy')
# plt.ylabel('Validation Accuracy')
# plt.xlabel('Epoch \nAVG accuracy : {:0.4f}; MAX accuracy : {:0.4f}; fc accuracy : {:0.4f}'.format(avg_2,max_2,fc_2))
# plt.legend(['AVG','MAX','FC'],loc="lower right")
# plt.grid(True)
# plt.show()

# plt.plot(avg_history.history['val_softmax1_loss'])
# plt.plot(max_history.history['val_softmax1_loss'])
# plt.plot(fc_history.history['val_softmax1_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['AVG','MAX','FC'],loc="upper right")
# plt.grid(True)
# plt.show()

# #EV
# plt.plot(avg_history.history['val_softmax3_accuracy'])
# plt.plot(max_history.history['val_softmax3_accuracy'])
# plt.plot(fc_history.history['val_softmax3_accuracy'])
# plt.title('Softmax3 accuracy')
# plt.ylabel('Validation Accuracy')
# plt.xlabel('Epoch \nAVG accuracy : {:0.4f}; MAX accuracy : {:0.4f}; fc accuracy : {:0.4f}'.format(avg_3,max_3,fc_3))
# plt.legend(['AVG','MAX','FC'],loc="lower right")
# plt.grid(True)
# plt.show()

# plt.plot(avg_history.history['val_softmax1_loss'])
# plt.plot(max_history.history['val_softmax1_loss'])
# plt.plot(fc_history.history['val_softmax1_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['AVG','MAX','FC'],loc="upper right")
# plt.grid(True)
# plt.show()

# print('Total : AVG accuracy : {:0.4f}; MAX accuracy : {:0.4f}; fc accuracy : {:0.4f}'.format(avg_fusion_acc,max_fusion_acc,fc_fusion_acc))
print(type(avg_fusion_acc))
# print(int('{}'.format(max_fusion_acc),0))
# print(int('{}'.format(fc_fusion_acc),0))

