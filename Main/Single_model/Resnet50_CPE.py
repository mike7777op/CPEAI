import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
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


X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/Y_test.npy')


# X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/train/X_train.npy')
# Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/train/Y_train.npy')
# X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/test/X_test.npy')
# Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/test/Y_test.npy')


print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None, 
    input_shape=(224,224,3),
    pooling=None,
    classes = 9
)

x = model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(9, activation='softmax', name='softmax')(x)


model_final = Model(inputs=model.input, outputs=x)
learning_rate = 0.00003
optimizer = Adam(learning_rate=learning_rate)
model_final.compile(optimizer=optimizer,
              loss=categorical_crossentropy,
              metrics=['accuracy'])
model_final.summary()

# filepath="/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/Balance_train/500/LR:0.00001/GMP_1/weights_best-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
training = model_final.fit(X_train, Y_train,validation_split=0.25,epochs=100, batch_size=32, shuffle=True)
# model_final.save('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/0716/InceptionV3/Single_model_1.h5')

preds = model_final.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#train loss,val loss
plt.plot(training.history['loss'],'r-.s')
plt.plot(training.history['val_loss'],'g--^')
plt.title('Training loss and val loss')
plt.ylabel("loss")`
plt.xlabel("epoch")
plt.legend(["Train_loss","Val_loss"],loc="upper right")`
plt.grid(True)
plt.show()

# print(training.history['loss'][99])
# print(training.history['val_loss'][99])

#y_label
axes = plt.axes()
axes.set_ylim([0.0,10])
plt.plot(training.history['loss'],'r-.s')
plt.plot(training.history['val_loss'],'g--^')
plt.title('Training loss and val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["Train_loss","Val_loss"],loc="upper right")
plt.grid(True)
plt.show()

#y_label
axes = plt.axes()
axes.set_ylim([0.0,5.0])
plt.plot(training.history['loss'],'r-.s')
plt.plot(training.history['val_loss'],'g--^')
plt.title('Training loss and val loss')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["Train_loss","Val_loss"],loc="upper right")
plt.grid(True)
plt.show()
# plt.plot(traning.history['accuracy'],':')
# plt.plot(traning.history['loss'])
# plt.title('models accuracy and loss')
# plt.grid(True)
# plt.show()

# plt.plot(traning.history['val_accuracy'])
# plt.plot(traning.history['val_loss'])
# plt.title('models val_accuracy and val_loss')
# plt.grid(True)
# plt.show()

plt.plot(training.history['accuracy'],'r-.^')
plt.plot(training.history['loss'],'g--')
plt.plot(training.history['val_accuracy'],'b--*')
plt.plot(training.history['val_loss'],'y-o')
plt.title("Resnet50_model_1")
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
plt.grid(True)
plt.show()

All_list = ['IA','IB','MDCK','Para1','Para2','Para3','MK2','EV','RD']
New_list = ['IA','IB','MDCK','Para1','Para2','Para3','MK2','EV','RD','RSV','Hep-2']
Influ_list = ['IA','IB','MDCK','None']
Para_list = ['Para1','Para2','Para3','MK2','None']
EV_list = ['EV','RD','None']
RSV_list = ['RSV','Hep-2','None']


# Confusion_marrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="black" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
    plt.show()

def plot_confusion(model,X_test,Y_test,labels):
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions,axis=1)
    truelabel = Y_test.argmax(axis=-1)  
    # conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)    
    cm = confusion_matrix(y_true=truelabel,y_pred=predictions)
    # plt.figure()
    plot_confusion_matrix(cm, normalize=True,target_names=labels,title='Confusion Matrix')
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Non normalize Confusion Matrix')


plot_confusion(model_final, X_test, Y_test,labels=All_list)

