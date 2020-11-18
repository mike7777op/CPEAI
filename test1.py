import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.resnet50 import ResNet50
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
classes1 = 9
classes2 = 3



X_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/train/X_train.npy')
Y_train1 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/train/Influ_Y_train1.npy')
Y_train2 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/train/Para_Y_train2.npy')
Y_train3 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/train/CB_Y_train3.npy')

X_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data_3/test/CB_Y_test3.npy')


print('X_train shape : ',X_train.shape)
print('Y_train1 shape : ',Y_train1.shape)
print('Y_train2 shape : ',Y_train2.shape)
print('Y_train3 shape : ',Y_train3.shape)

print('X_test shape : ',X_test.shape)
print('Y_test1 shape : ',Y_test1.shape)
print('Y_test2 shape : ',Y_test2.shape)
print('Y_test3 shape : ',Y_test3.shape)



model1 = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None, 
    input_shape=(224,224,3),
    pooling='max',
    classes = [4,5,3]
)

x1 = model1.output
x1 = Dropout(0.5)(x1)
x2 = Dropout(0.5)(x1)
x3 = Dropout(0.5)(x1)
# MDCK = Dense(3, activation='softmax', name = 'softmax1')(x1)
# MK2 = Dense(4, activation='softmax', name = 'softmax2')(x1)
# RD = Dense(2,activation='softmax',name = 'softmax3')(x1)
Influ = Dense(4,activation='softmax', name='softmax1')(x1)
Para = Dense(5, activation='softmax', name='softmax2')(x2)
CB = Dense(3, activation='softmax', name='softmax3')(x3)
model_final = Model(inputs=model1.input,outputs=[Influ,Para,CB])
# opt = keras.optimizers.Adam(learning_rate=0.00001)
model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
            loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy},
            metrics=['accuracy'])
model_final.summary()


traning = model_final.fit(X_train,[Y_train1,Y_train2,Y_train3],validation_split=0.2,epochs=100, batch_size=32, shuffle=True)
model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/IPC_Mutitask_three_.h5')

loss, softmax1_loss, softmax2_loss, softmax3_loss, softmax1_acc, softmax2_acc, softmax3_acc = model_final.evaluate(X_test, [Y_test1,Y_test2,Y_test3])
print ("Softmax1 Loss = " + str(softmax1_loss))
print ("Softmax2 Loss = " + str(softmax2_loss))
print ("Softmax3 Loss = " + str(softmax3_loss))

print ("Softmax1 Accuracy = " + str(softmax1_acc))
print ("Softmax2 Accuracy = " + str(softmax2_acc))
print ("Softmax3 Accuracy = " + str(softmax3_acc))


plt.plot(traning.history['softmax1_accuracy'])
plt.plot(traning.history['softmax1_loss'])
plt.title('softmax1 accuracy and loss')
plt.show()

plt.plot(traning.history['val_softmax1_accuracy'])
plt.plot(traning.history['val_softmax1_loss'])
plt.title('softmax1_val_accuracy and val_loss')
plt.show()

plt.plot(traning.history['softmax2_accuracy'])
plt.plot(traning.history['softmax2_loss'])
plt.title('softmax2 accuracy and loss')
plt.show()

plt.plot(traning.history['val_softmax2_accuracy'])
plt.plot(traning.history['val_softmax2_loss'])
plt.title('softmax2_val_accuracy and val_loss')
plt.show()

plt.plot(traning.history['softmax3_accuracy'])
plt.plot(traning.history['softmax3_loss'])
plt.title('softmax3 accuracy and loss')
plt.show()

plt.plot(traning.history['val_softmax3_accuracy'])
plt.plot(traning.history['val_softmax3_loss'])
plt.title('softmax3_val_accuracy and val_loss')
plt.show()


Influ_list = ['IA','IB','MDCK','None']
Para_list = ['Para1','Para2','Para3','MK2','None']
CB1RD_list = ['CB1','RD','None']

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 10))
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
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


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
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Confusion Matrix')

plot_confusion(model_final, X_test, Y_test1,labels=Influ_list,index=0)
plot_confusion(model_final, X_test, Y_test2,labels=Para_list,index=1)
plot_confusion(model_final, X_test, Y_test3,labels=CB1RD_list,index=2)