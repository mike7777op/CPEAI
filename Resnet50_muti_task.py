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
from Resnet50_package.Resnet50_2 import Resnet50_2 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

classes1 = 9
classes2 = 3


X_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/train/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/train/Y_train.npy')
y_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/train/y_train.npy')
X_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/test/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/test/Y_test.npy')
y_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/CV_data/test/y_test.npy')

print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('y_train shape : ',y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)
print('y_test shape : ',y_test.shape)



# def model1():
def Resnet50_model1(classes):
    model1 = ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None, 
        input_shape=(224,224,3),
        pooling='max',
        classes = classes
    )

    x1 = model1.output
    # x1 = Flatten(name = 'flatten1')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(classes, activation='softmax', name='softmax1')(x1)
    # # for layer in model_final.layers[:FREEZE_LAYERS]:
    # #     layer.trainable = False
    # # for layer in model_final.layers[FREEZE_LAYERS:]:
    # #     layer.trainable = True
    
    return x1,model1.input

def Resnet50_model2(classes):
    model2 = Resnet50_2(
        include_top=False,
        weights='imagenet',
        input_tensor=None, 
        input_shape=(224,224,3),
        pooling='max',
        classes = classes
    )

    x2 = model2.output
    # x2 = Flatten(name = 'flatten2')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(classes, activation='softmax', name='softmax2')(x2)
    # for layer in model_final.layers[:FREEZE_LAYERS]:
    #     layer.trainable = False
    # for layer in model_final.layers[FREEZE_LAYERS:]:
    #     layer.trainable = True
    
    return x2,model2.input

def build_model(classes1,classes2):
    x1, y1 = Resnet50_model1(classes=classes1)
    x2, y2 = Resnet50_model2(classes=classes2)

    model_final = Model(inputs=[y1, y2],outputs=[x1, x2])
    # opt = keras.optimizers.Adam(learning_rate=0.00001)
    model_final.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.00001),
              loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy},
              metrics=['accuracy'])
    model_final.summary()
    # traning = model_final.fit([X_train,X_train],[Y_train,y_train2],validation_split=0.2,epochs=100, batch_size=32, shuffle=True)
#     # model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Para1MK2_DataGenerator_imagenet_resnet_model_1.h5')
#     model_final.save('/home/pmcn/workspace/Test_Code/Resnet50/checkpoint/Mutitask_3.h5')
#     plt.plot(traning.history['softmax1_accuracy'])
#     plt.plot(traning.history['softmax1_loss'])
#     plt.title('softmax1 accuracy and loss')
#     plt.show()

#     plt.plot(traning.history['softmax2_accuracy'])
#     plt.plot(traning.history['softmax2_loss'])
#     plt.title('softmax2 accuracy and loss')
#     plt.show()

#     plt.plot(traning.history['val_softmax1_accuracy'])
#     plt.plot(traning.history['val_softmax1_loss'])
#     plt.title('softmax1 val_accuracy and val_loss')
#     plt.show()

#     plt.plot(traning.history['val_softmax2_accuracy'])
#     plt.plot(traning.history['val_softmax2_loss'])
#     plt.title('softmax2 val_accuracy and val_loss')
#     plt.show()




if __name__ == '__main__':
    # model = Resnet50_model1(classes=classes1)
    # model2 = Resnet50_model2(classes=classes2)
    build = build_model(classes1=classes1,classes2=classes2)
    


# labels = ['CB1','IA','IB','Para1','Para2','Para3','MDCK','MK2','RD']
# # labels = ['MDCK','MK2','RD']

# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
#                           normalize=True):
   
 
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(15, 10))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
# 	#plt.savefig('/content/drive/My Drive/Colab Notebooks/confusionmatrix32.png',dpi=350)
#     plt.show()

# def plot_confusion(model,X_test,Y_test):
#     predictions = model.predict(X_test)
#     predictions = np.argmax(predictions,axis=1)
#     truelabel = Y_test.argmax(axis=-1)   # 将one-hot转化为label
#     # conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)    
#     cm = confusion_matrix(y_true=truelabel,y_pred=predictions)
#     plt.figure()
#     plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Confusion Matrix')

# plot_confusion(model_final, X_test, Y_test)