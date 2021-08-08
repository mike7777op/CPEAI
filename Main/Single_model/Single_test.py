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
from sklearn.metrics import roc_curve,classification_report
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix
import itertools
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)


All_list = {0:'IA',1:'IB',2:'MDCK',3:'Para1',4:'Para2',5:'Para3',6:'MK2',7:'EV',8:'RD'}
_11_list = {0:'IA',1:'IB',2:'MDCK',3:'Para1',4:'Para2',5:'Para3',6:'MK2',7:'EV',8:'RD',9:'RSV',10:'Hep-2'}
_13_list = {0:'IA',1:'IB',2:'MDCK',3:'Para1',4:'Para2',5:'Para3',6:'MK2',7:'EV',8:'RD',9:'RSV',10:'Hep-2',11:'ADV',12:'A549'}
Influ_list = {0:'IA',1:'IB',2:'MDCK',3:'None'}
Para_list = {0:'Para1',1:'Para2',2:'Para3',3:'MK2',4:'None'}
EV_list = {0:'CB1',1:'RD',2:'None'}
RSV_list = {0:'RSV',1:'Hep-2'}
ADV_list = {0:'ADV',1:'A549'}

#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/More_task/300/9_class/weights_best-88_0.9444.hdf5')
# model = load_model('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Single/Single_All_1.h5')

# X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/test/X_test.npy')
# Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/MK2_task/Singel/test/Y_test.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/9_class/test/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/More_task/300/9_class/test/EV_Y_test3.npy')

nb_classes = Y_test.shape[1]

#test_img
img_path = '/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/original/test/IA_500/404.jpg'

# img = image.load_img(img_path, target_size=(224, 224))

# # plt.imshow(img)
# img = image.img_to_array(img) / 255.0
# img = np.expand_dims(img, axis=0)  
# # pred = model.predict(img)
# pred1 = model.predict(img)[0]
# pred2 = model.predict(img)[1]
# pred3 = model.predict(img)[2]

# # for i in pred:
# #     top_inds = i.argsort()[::-1][:5]
# #     s = top_inds
# #     # print(top_inds)
# #     t = i[s]
# #     print(t)
# #     for j in top_inds:
# #         print('{:.3f}  {}'.format(i[j],All_list[j]))

# # Multi task learning
# for i,j,k in zip(pred1,pred2,pred3):
#     top_inds1 = i.argsort()[::-1][:5]
#     s1 = top_inds1
#     t1 = i[s1]
#     print('MDCK predict:',t1)
    
#     top_inds2 = j.argsort()[::-1][:5]
#     s2 = top_inds2 
#     t2 = j[s2]
#     print('MK2 predict:',t2)

#     top_inds3 = k.argsort()[::-1][:5]
#     s3 = top_inds3
#     t3 = k[s3]
#     print('RD predict:',t3)

#(x_test)[]---.h5擁有多個weight
pred = model.predict(X_test)[2]
predictions = np.argmax(pred,axis=1)
truelabel = Y_test.argmax(axis=-1) 

print(predictions.shape)

acc = accuracy_score(truelabel,predictions,normalize=True)
print("Accuracy: {:.2f}%".format(acc*100))

precision = precision_score(truelabel, predictions,average='micro')
recall = recall_score(truelabel, predictions, average='micro')
f1_score = f1_score(truelabel, predictions, average='micro')

print('Precision score:',precision)
print('Recall score:',recall)
print('F1_score:',f1_score)

print(classification_report(truelabel, predictions ,target_names=EV_list,digits=4))

# Confusion_matrix
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
    predictions = model.predict(X_test)[2]
    predictions = np.argmax(predictions,axis=1)
    truelabel = Y_test.argmax(axis=-1)  
    # conf_mat = confusion_matrix(y_true=truelabel, predictions=predictions)    
    cm = confusion_matrix(y_true=truelabel,y_pred=predictions)
    # plt.figure()
    plot_confusion_matrix(cm, normalize=True,target_names=labels,title='Confusion Matrix')
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Non normalize Confusion Matrix')


plot_confusion(model, X_test, Y_test,labels=EV_list)
