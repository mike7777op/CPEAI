import os,sys
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import random
from random import choice
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

All_list = ['IA','IB','MDCK','Para1','Para2','Para3','MK2','CB1','RD']
Influ_list = {0:'IA',1:'IB',2:'MDCK',3:'None'}
Para_list = {0:'Para1',1:'Para2',2:'Para3',3:'MK2',4:'None'}
EVRD_list = {0:'CB1',1:'RD',2:'None'}


#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint2/0121_Paraless_Multi_model_1.h5')

X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Para_less_data/test_npy/Multi/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Para_less_data/test_npy/Multi/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Para_less_data/test_npy/Multi/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Para_less_data/test_npy/Multi/CB_Y_test3.npy')



# plt.imshow(img)
# pred = model.predict(X_test)
# # predictions = np.argmax(pred,axis=1)
# truelabel = Y_test.argmax(axis=-1) 

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

def plot_confusion(model,X_test,Y_test,labels,index):
    predictions = model.predict(X_test)[index]
    predictions = np.argmax(predictions,axis=1)
    truelabel = Y_test.argmax(axis=-1)  
    # conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)    
    cm = confusion_matrix(y_true=truelabel,y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(cm, normalize=True,target_names=labels,title='Confusion Matrix')
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Non normalize Confusion Matrix')

plot_confusion(model, X_test, Y_test1,labels=Influ_list,index=0)
plot_confusion(model, X_test, Y_test2,labels=Para_list,index=1)
plot_confusion(model, X_test, Y_test3,labels=EVRD_list,index=2)