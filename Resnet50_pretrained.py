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



X_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CIP_Generator/test_npy/Y_test.npy')

print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None, 
    input_shape=(224,224,3),
    pooling='max',
    classes = 9
)
x = model.output
x = Dropout(0.5)(x)
x = Dense(9, activation='softmax', name='softmax')(x)

model_final = Model(inputs=model.input, outputs=x)
# for layer in model_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in model_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True

model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
              loss=categorical_crossentropy,
              metrics=['accuracy'])

model_final.summary()
traning = model_final.fit(X_train, Y_train, validation_split=0.2 ,epochs=100, batch_size=32, shuffle=True)
# model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Para1MK2_DataGenerator_imagenet_resnet_model_1.h5')
model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Single_All_1.h5')

preds = model_final.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

plt.plot(traning.history['accuracy'])
plt.plot(traning.history['loss'])
plt.title('models accuracy and loss')
plt.show()

plt.plot(traning.history['val_accuracy'])
plt.plot(traning.history['val_loss'])
plt.title('models val_accuracy and val_loss')
plt.show()

Influ_list = ['IA','IB','MDCK','None']
Para_list = ['Para1','Para2','Para3','MK2','None']
CB1RD_list = ['CB1','RD','None']
All_list = ['CB1','IA','IB','Para1','Para2','Para3','MDCK','MK2','RD']

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

def plot_confusion(model,X_test,Y_test,labels):
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions,axis=1)
    truelabel = Y_test.argmax(axis=-1)  
    # conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)    
    cm = confusion_matrix(y_true=truelabel,y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Confusion Matrix')

plot_confusion(model_final, X_test, Y_test,labels=All_list)