import os,sys,io
import numpy as np
import scipy
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.vgg16 import VGG16
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
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)



#數據
# X_train = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/train_npy/Multi/X_train.npy')
# Y_train1 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/train_npy/Multi/Influ_Y_train1.npy')
# Y_train2 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/train_npy/Multi/Para_Y_train2.npy')
# Y_train3 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/train_npy/Multi/CB_Y_train3.npy')

# X_test = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/test_npy/Multi/X_test.npy')
# Y_test1 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/test_npy/Multi/Influ_Y_test1.npy')
# Y_test2 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/test_npy/Multi/Para_Y_test2.npy')
# Y_test3 = np.load('/home/pmcn/workspace/Test_Code/Resnet50/Para_less_data/test_npy/Multi/CB_Y_test3.npy')

X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/train_npy/X_train.npy')
Y_train1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/train_npy/Influ_Y_train1.npy')
Y_train2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/train_npy/Para_Y_train2.npy')
Y_train3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/train_npy/EV_Y_train3.npy')
# Y_train4 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator_0114/train_npy/Multi_2/RSV_Y_train4.npy')
# Y_train5 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_CIP_Generator_0114/train_npy/Multi/ADV_Y_train5.npy')

X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/test_npy/X_test.npy')
Y_test1 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/test_npy/Influ_Y_test1.npy')
Y_test2 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/test_npy/Para_Y_test2.npy')
Y_test3 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/0216_data/100/Multi/test_npy/EV_Y_test3.npy')
# Y_test4 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator_0114/test_npy/Multi_2/RSV_Y_test4.npy')
# Y_test5 = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_CIP_Generator_0114/test_npy/Multi/ADV_Y_test5.npy')



print('X_train shape : ',X_train.shape)
print('Y_train1 shape : ',Y_train1.shape)
print('Y_train2 shape : ',Y_train2.shape)
print('Y_train3 shape : ',Y_train3.shape)
# print('Y_train4 shape : ',Y_train4.shape)
# print('Y_train5 shape : ',Y_train5.shape)

print('X_test shape : ',X_test.shape)
print('Y_test1 shape : ',Y_test1.shape)
print('Y_test2 shape : ',Y_test2.shape)
print('Y_test3 shape : ',Y_test3.shape)
# print('Y_test4 shape : ',Y_test4.shape)
# print('Y_test5 shape : ',Y_test5.shape)


#Resnet50
model1 = VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None, 
    input_shape=(224,224,3),
    pooling='avg',
    classes = [4,5,3]
)

x1 = model1.output
#使用Droupout
x1 = Dropout(0.5)(x1)
#Multi tasking
Influ = Dense(4,activation='softmax', name='softmax1')(x1)
Para = Dense(5, activation='softmax', name='softmax2')(x1)
EV = Dense(3, activation='softmax', name='softmax3')(x1)
# RSV = Dense(3, activation='softmax', name='softmax4')(x1)
# ADV = Dense(3, activation='softmax', name='softmax5')(x1)

# model_final = keras.models.load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint2/0120_all_Multi_model_1.h5')

model_final = Model(inputs=model1.input,outputs=[Influ,Para,EV])

learning_rate = 0.00001 
optimizer = Adam(lr=learning_rate)
model_final.compile(optimizer=optimizer,
            # loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy, 'softmax4':categorical_crossentropy},
            loss={'softmax1':categorical_crossentropy,'softmax2':categorical_crossentropy,'softmax3':categorical_crossentropy},
            metrics=['accuracy'])
model_final.summary()

filepath="/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/VGG_2/Multi_100/weights_best-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
csvlogger = keras.callbacks.CSVLogger(filename='VGG16_paraless_Multi.csv',separator=',',append=True)
traning = model_final.fit(X_train,[Y_train1,Y_train2,Y_train3] ,epochs=150, batch_size=32, shuffle=True,callbacks=[checkpoint,csvlogger])
model_final.save('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/VGG_2/Multi_100/0217_Multi_model_1.h5')

# loss, softmax1_loss, softmax2_loss, softmax3_loss, softmax4_loss, softmax1_acc, softmax2_acc, softmax3_acc, softmax4_acc = model_final.evaluate(X_test, [Y_test1,Y_test2,Y_test3,Y_test4])
loss, softmax1_loss, softmax2_loss, softmax3_loss, softmax1_acc, softmax2_acc, softmax3_acc  = model_final.evaluate(X_test, [Y_test1,Y_test2,Y_test3])

#Test loss and acc
print('loss = ' + str(loss))

print ("Softmax1 Loss = " + str(softmax1_loss))
print ("Softmax2 Loss = " + str(softmax2_loss))
print ("Softmax3 Loss = " + str(softmax3_loss))
# print('Softmax4 Loss = '  +str(softmax4_loss))
# print('Softmax5 Loss = ' + str(softmax5_loss))

print ("Softmax1 Accuracy = " + str(softmax1_acc))
print ("Softmax2 Accuracy = " + str(softmax2_acc))
print ("Softmax3 Accuracy = " + str(softmax3_acc))
# print('Softmax4 Accuracy = ' + str(softmax4_acc))
# print('Softmax5_Accuracy = ' + str(softmax5_acc))

#train loss,val loss
# plt.plot(traning.history['loss'],':')
# plt.plot(traning.history['val_loss'],'--')
# plt.title('Training loss and val loss')
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend(["Train_loss","Val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

#Influ loss and acc
plt.plot(traning.history['softmax1_accuracy'])
plt.plot(traning.history['softmax1_loss'])
plt.title('softmax1 accuracy and loss')
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_acc","train_loss"],loc="upper left")
plt.grid(True)
plt.show()

#Influ val_loss and val_acc
# plt.plot(traning.history['val_softmax1_accuracy'])
# plt.plot(traning.history['val_softmax1_loss'])
# plt.title('softmax1_val_accuracy and val_loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

# plt.plot(traning.history['softmax1_accuracy'])
# plt.plot(traning.history['softmax1_loss'])
# plt.plot(traning.history['val_softmax1_accuracy'])
# plt.plot(traning.history['val_softmax1_loss'])
# plt.title('softmax1')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

#Para loss and acc
plt.plot(traning.history['softmax2_accuracy'])
plt.plot(traning.history['softmax2_loss'])
plt.title('softmax2 accuracy and loss')
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_acc","train_loss"],loc="upper left")
plt.grid(True)
plt.show()

#Para val_loss and val_acc
# plt.plot(traning.history['val_softmax2_accuracy'])
# plt.plot(traning.history['val_softmax2_loss'])
# plt.title('softmax2_val_accuracy and val_loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

# plt.plot(traning.history['softmax2_accuracy'])
# plt.plot(traning.history['softmax2_loss'])
# plt.plot(traning.history['val_softmax2_accuracy'])
# plt.plot(traning.history['val_softmax2_loss'])
# plt.title('softmax2')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

#EV loss and acc
plt.plot(traning.history['softmax3_accuracy'])
plt.plot(traning.history['softmax3_loss'])
plt.title('softmax3 accuracy and loss')
plt.ylabel("loss/acc")
plt.xlabel("epoch")
plt.legend(["train_acc","train_loss"],loc="upper left")
plt.grid(True)
plt.show()

#EV val_loss and val_acc
# plt.plot(traning.history['val_softmax3_accuracy'])
# plt.plot(traning.history['val_softmax3_loss'])
# plt.title('softmax3_val_accuracy and val_loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()

# plt.plot(traning.history['softmax3_accuracy'])
# plt.plot(traning.history['softmax3_loss'])
# plt.plot(traning.history['val_softmax3_accuracy'])
# plt.plot(traning.history['val_softmax3_loss'])
# plt.title('softmax3')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.grid(True)
# plt.show()


#RSV loss and acc
# plt.plot(traning.history['softmax4_accuracy'])
# plt.plot(traning.history['softmax4_loss'])
# plt.title('softmax4 accuracy and loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss"],loc="upper left")
# plt.show()

# #RSV val_loss and val_acc
# plt.plot(traning.history['val_softmax4_accuracy'])
# plt.plot(traning.history['val_softmax4_loss'])
# plt.title('softmax4_val_accuracy and val_loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["val_acc","val_loss"],loc="upper left")
# plt.show()

# plt.plot(traning.history['softmax4_accuracy'])
# plt.plot(traning.history['softmax4_loss'])
# plt.plot(traning.history['val_softmax4_accuracy'])
# plt.plot(traning.history['val_softmax4_loss'])
# plt.title('softmax4')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.show()

# #ADV loss and acc
# plt.plot(traning.history['softmax5_accuracy'])
# plt.plot(traning.history['softmax5_loss'])
# plt.title('softmax5 accuracy and loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss"],loc="upper left")
# plt.show()

# #ADV val_loss and val_acc
# plt.plot(traning.history['val_softmax5_accuracy'])
# plt.plot(traning.history['val_softmax5_loss'])
# plt.title('softmax5_val_accuracy and val_loss')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["val_acc","val_loss"],loc="upper left")
# plt.show()

# plt.plot(traning.history['softmax5_accuracy'])
# plt.plot(traning.history['softmax5_loss'])
# plt.plot(traning.history['val_softmax5_accuracy'])
# plt.plot(traning.history['val_softmax5_loss'])
# plt.title('softmax5')
# plt.ylabel("loss/acc")
# plt.xlabel("epoch")
# plt.legend(["train_acc","train_loss","val_acc","val_loss"],loc="upper left")
# plt.show()




Influ_list = ['IA','IB','MDCK','None']
Para_list = ['Para1','Para2','Para3','MK2','None']
EVRD_list = ['EV','RD','None']
RSVHep_list = ['RSV','Hep-2','None']
ADVA549_list = ['ADV','A549','None']


#Confusion Matrix
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
    # plt.figure()
    plot_confusion_matrix(cm, normalize=True,target_names=labels,title='Confusion Matrix')
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Non normalize Confusion Matrix')

plot_confusion(model_final, X_test, Y_test1,labels=Influ_list,index=0)
plot_confusion(model_final, X_test, Y_test2,labels=Para_list,index=1)
plot_confusion(model_final, X_test, Y_test3,labels=EVRD_list,index=2)
# plot_confusion(model_final, X_test, Y_test4,labels=RSVHep_list,index=3)
# plot_confusion(model_final, X_test, Y_test5,labels=ADVA549_list,index=4)


# visualkeras.layered_view(model_final).show()
