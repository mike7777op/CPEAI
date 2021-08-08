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
EV_list = {0:'CB1',1:'RD',2:'None'}


#使用pre-trained model .h5
model = load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint2/0115_IPER_Multi_model_3.h5')

X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/IPER_0115/Single/test/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/IPER_0115/Single/test/Y_test.npy')




# plt.imshow(img)

#三個分數比大小
def score1(pred1,pred2,pred3):
    for j,k,l in zip(pred1,pred2,pred3):
        top_inds1 = j.argsort()[::-1][:5]
        top_inds2 = k.argsort()[::-1][:5]
        top_inds3 = l.argsort()[::-1][:5]
        s1 = top_inds1[0]
        s2 = top_inds2[0]
        s3 = top_inds3[0]
        if s1>s2:
            if s2>s3:
                # print('s1>s2>s3')
                return 1
            elif s2 == s3:
                # print('s1>s2=s3')
                return 1
            elif s2<s3:
                if s1<s3:
                    # print('s3>s1>s2')
                    return 3
                elif s1 == s3:
                    # print('s3=s1>2')
                    return 31
                elif s1>s3:
                    # print('s1>s3>s2')
                    return 1
        
        elif s1 == s2:
            if s2>s3:
                # print('s1=s2>s3')
                return 12
            elif s2 == s3:
                # print('s1=s2=s3')
                return 123
            else:
                if s1<s3:
                    # print('s3>s1=s2')
                    return 3
                elif s1==s3:
                    # print('s1=s2=s3')
                    return 123
                elif s1>s3:
                    # print('s1=s2>s3')
                    return 12
        elif s1<s2:
            if s2<s3:
                # print('s3>s2>s1')
                return 3
            else:
                if s2 == s3:
                    # print('s2=s3>s1')
                    return 23
                else:
                    if s1<s3:
                        # print('s2>s3>s1')
                        return 2
                    elif s1 == s3:
                        # print('s2>s3>s1')
                        return 2
                    elif s1>s3:
                        # print('s2>s1>s3')
                        return 2
#兩個分數比大小
def score2(pred1,pred2):
    for j,k in zip(pred1,pred2):
        top_inds1 = j.argsort()[::-1][:5]
        top_inds2 = k.argsort()[::-1][:5]
        s1 = top_inds1[0]
        s2 = top_inds2[0]
        if s1>s2:
            # print('s1')
            return 1
        elif s1 == s2:
            # print('s1=s2')
            return 12
        elif s1<s2:
            # print('s2')
            return 2

def score3(pred1,pred2,pred3,pred4):
    for j,k,l,M in zip(pred1,pred2,pred3):
        top_inds1 = j.argsort()[::-1][:5]
        top_inds2 = k.argsort()[::-1][:5]
        top_inds3 = l.argsort()[::-1][:5]
        top_inds4 = M.argsort()[::-1][:5]
        s1 = top_inds1[0]
        s2 = top_inds2[0]
        s3 = top_inds3[0]
        s4 = top_inds4[0]
        
        return 1


def fusion(X_test,Y_test):

    pred1 = model.predict(X_test)[0]
    pred2 = model.predict(X_test)[1]
    pred3 = model.predict(X_test)[2]
    pred4 = model.predict(X_test)[3]
    predictions1 = np.argmax(pred1,axis=1)
    predictions2 = np.argmax(pred2,axis=1)
    predictions3 = np.argmax(pred3,axis=1)
    predictions4 = np.argmax(pred4,axis=1)

    truelabel = Y_test.argmax(axis=-1)  
    
  

    # for i in predictions1:
    #     print(Influ_list(i))
    
    # for i in pred1:
    #     top_inds = i.argsort()[::-1][:5]
        
    #     for j in top_inds:
    #         print('    {:.3f}  {}'.format(i[j], Influ_list[j]))
    length = len(predictions3)
    # print(length)
    new_pred = []


    i = 0
    while i < length:
        p1 = pred1[i]
        p2 = pred2[i]
        p3 = pred3[i]
        p4 = pred4[i]
        #a,b,c位置
        a = predictions1[i]
        b = predictions2[i]
        c = predictions3[i]
        d = predictions4[i]

        if a == 0:
            if b == 0:
                if c == 0:
                    if d == 0:
                        print('a0 vs b0 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [0,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,3,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b0 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,3,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b0 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [0,3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a0 vs b0 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [0,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,3,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs b0 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,3,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b0 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [0,3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a0 vs b0 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b0 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b0')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        else:
                            l = [0,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 1:
                if c == 0:
                    if d == 0:
                        print('a0 vs b1 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [0,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,4,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b1 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,4,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b1 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [0,4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a0 vs b1 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [0,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,4,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs b1 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,4,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b1 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [0,4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a0 vs b1 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b1 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b1')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        else:
                            l = [0,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 2:
                if c == 0:
                    if d == 0:
                        print('a0 vs b2 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [0,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,5,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b2 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,5,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b2 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [0,5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a0 vs b2 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [0,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,5,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs b2 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,5,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b2 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [0,5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a0 vs b2 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b2 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b2')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        else:
                            l = [0,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    if d == 0:
                        print('a0 vs b3 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [0,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,6,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b3 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,6,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b3 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [0,6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a0 vs b3 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [0,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,6,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs b3 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [0,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [0,6,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b3 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [0,6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a0 vs b3 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a0 vs b3 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs b3')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        else:
                            l = [0,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 4:
                if c == 0:
                    if d == 0:
                        print('a0 vs c0 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs c0 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs c0')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[0,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('a0 vs c1 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs c1 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0 vs c1')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[0,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('a0 vs d0')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [0,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a0 vs d1')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a0')
                            new_pred.append(0)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [0,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a0')
                        new_pred.append(0)

        elif a == 1:
            if b == 0:
                if c == 0:
                    if d == 0:
                        print('a1 vs b0 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [1,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,3,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b0 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,3,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b0 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [1,3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a1 vs b0 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [1,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,3,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs b0 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,3,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b0 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [1,3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a1 vs b0 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b0 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b0')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        else:
                            l = [1,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 1:
                if c == 0:
                    if d == 0:
                        print('a1 vs b1 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [1,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,4,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b1 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,4,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b1 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [1,4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a1 vs b1 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [1,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,4,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs b1 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,4,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b1 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [1,4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a1 vs b1 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b1 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b1')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        else:
                            l = [1,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 2:
                if c == 0:
                    if d == 0:
                        print('a1 vs b2 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [1,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,5,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b2 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,5,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b2 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [1,5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a1 vs b2 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [1,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,5,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs b2 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,5,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b2 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [1,5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a1 vs b2 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b2 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b2')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        else:
                            l = [1,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    if d == 0:
                        print('a1 vs b3 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [1,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,6,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b3 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,6,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b3 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [1,6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a1 vs b3 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [1,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,6,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs b3 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [1,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [1,6,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b3 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [1,6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a1 vs b3 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a1 vs b3 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs b3')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        else:
                            l = [1,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 4:
                if c == 0:
                    if d == 0:
                        print('a1 vs c0 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs c0 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs c0')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[1,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('a1 vs c1 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs c1 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1 vs c1')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[1,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('a1 vs d0')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [1,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a1 vs d1')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a1')
                            new_pred.append(1)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [1,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a1')
                        new_pred.append(1)

        elif  a == 2:
            if b == 0:
                if c == 0:
                    if d == 0:
                        print('a2 vs b0 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [2,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,3,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a2 vs b0 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,3,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b0 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [2,3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a2 vs b0 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [2,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,3,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs b0 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,3,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b0 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [2,3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a2 vs b0 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a2 vs b0 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(0)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b0')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b0')
                            new_pred.append(3)
                        else:
                            l = [2,3]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 1:
                if c == 0:
                    if d == 0:
                        print('a2 vs b1 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [2,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,4,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append()

                    elif d == 1:
                        print('a2 vs b1 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,4,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b1 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [2,4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a2 vs b1 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [2,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,4,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs b1 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,4,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b1 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [2,4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a2 vs b1 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a2 vs b1 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b1')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b1')
                            new_pred.append(4)
                        else:
                            l = [2,4]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 2:
                if c == 0:
                    if d == 0:
                        print('a2 vs b2 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [2,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,5,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append()

                    elif d == 1:
                        print('a2 vs b2 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,5,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b2 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [2,5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a2 vs b2 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [2,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,5,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs b2 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,5,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b2 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [2,5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a2 vs b2 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a2 vs b2 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b2')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b2')
                            new_pred.append(5)
                        else:
                            l = [2,5]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    if d == 0:
                        print('a2 vs b3 vs c0 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                        elif ans == 12:
                            l = [2,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,6,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append()

                    elif d == 1:
                        print('a2 vs b3 vs c0 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,6,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b3 vs c0')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l = [2,6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                elif c == 1:
                    if d == 0:
                        print('a2 vs b3 vs c1 vs d0')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d0')
                            new_pred.append(9)
                            elif ans == 12:
                            l = [2,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,6,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs b3 vs c1 vs d1')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 4:
                            print('d1')
                            new_pred.append(10)
                        elif ans == 12:
                            l = [2,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 13:
                            l = [2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 14:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 23:
                            l = [6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 24:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 34:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            l = [2,6,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b3 vs c1')
                        ans = score1(p1,p2,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l = [2,6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
    
                elif c == 2:
                    if d == 0:
                        print('a2 vs b3 vs d0')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('a2 vs b3 vs d1')
                        ans = score1(p1,p2,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs b3')
                        ans = score2(p1,p2)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('b3')
                            new_pred.append(6)
                        else:
                            l = [2,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

            elif b == 4:
                if c == 0:
                    if d == 0:
                        print('a2 vs c0 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs c0 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs c0')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[2,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('a2 vs c1 vs d0')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs c1 vs d1')
                        ans = score1(p1,p3,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2 vs c1')
                        ans = score2(p1,p3)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[2,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('a2 vs d0')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [2,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('a2 vs d1')
                        ans = score2(p1,p4)
                        if ans == 1:
                            print('a2')
                            new_pred.append(2)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [2,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('a2')
                        new_pred.append(2)
        
        elif a == 3:
            if b == 0:
                if c == 0:
                    if d == 0:
                        print('b0 vs c0 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [3,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b0 vs c0 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [3,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        # print('b0 vs c0')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[3,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('b0 vs c1 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [3,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b0 vs c1 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [3,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b0 vs c1')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[3,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('b0 vs d0')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [3,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('b0 vs d1')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b0')
                            new_pred.append(3)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [3,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('b0')
                        new_pred.append(3)

            elif b == 1:
                if c == 0:
                    if d == 0:
                        print('b1 vs c0 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [4,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b4 vs c0 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b4')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [4,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b1 vs c0')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[4,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('b1 vs c1 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [4,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b1 vs c1 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [4,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b1 vs c1')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[4,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('b1 vs d0')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [4,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('b1 vs d1')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b1')
                            new_pred.append(4)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [4,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('b1')
                        new_pred.append(4)

            elif b == 2:
                if c == 0:
                    if d == 0:
                        print('b2 vs c0 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [5,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b2 vs c0 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [5,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b2 vs c0')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[5,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('b2 vs c1 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [5,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b2 vs c1 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [5,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b2 vs c1')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[5,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('b2 vs d0')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [5,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('b2 vs d1')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b2')
                            new_pred.append(5)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [5,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('b2')
                        new_pred.append(5)
            
            elif b == 3:
                if c == 0:
                    if d == 0:
                        print('b3 vs c0 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [6,7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b3 vs c0 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [6,7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b3 vs c0')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c0')
                            new_pred.append(7)
                        else:
                            l=[6,7]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 1:
                    if d == 0:
                        print('b3 vs c1 vs d0')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [6,8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('b3 vs c1 vs d1')
                        ans = score1(p2,p3,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 3:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [6,8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d ==2:
                        print('b3 vs c1')
                        ans = score2(p2,p3)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('c1')
                            new_pred.append(8)
                        else:
                            l=[6,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                elif c == 2:
                    if d == 0:
                        print('b3 vs d0')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [6,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)

                    elif d == 1:
                        print('b3 vs d1')
                        ans == score2(p2,p4)
                        if ans == 1:
                            print('b3')
                            new_pred.append(6)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [6,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:
                        print('b3')
                        new_pred.append(6)

            elif b == 4:
                if c == 0:
                    if d == 0:
                        print('c0 vs d0')
                        ans == score2(p3,p4)
                        if ans == 1:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [7,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('c0 vs d1')
                        ans == score2(p3,p4)
                        if ans == 1:
                            print('c0')
                            new_pred.append(7)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [7,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:    
                        print('c0')
                        new_pred.append(7)
                elif c == 1:
                    if d == 0:
                        print('c1 vs d0')
                        ans == score2(p3,p4)
                        if ans == 1:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 2:
                            print('d0')
                            new_pred.append(9)
                        else:
                            l = [8,9]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 1:
                        print('c1 vs d1')
                        ans == score2(p3,p4)
                        if ans == 1:
                            print('c1')
                            new_pred.append(8)
                        elif ans == 2:
                            print('d1')
                            new_pred.append(10)
                        else:
                            l = [8,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                    elif d == 2:    
                        print('c1')
                        new_pred.append(8)
                elif c == 2:
                    if d == 0:
                        print('d0')
                        new_pred.append(9)
                    elif d == 1:
                        print('d1')
                        new_pred.append(10)
                    elif d ==2:
                        print('None')
                        ans = score3(p1,p2,p3,p4)
                        if ans == 1:
                            l = [0,1,2]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 2:
                            l = [3,4,5,6]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 3:
                            l = [7,8]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        elif ans == 4:
                            l = [9,10]
                            r = choice(l)
                            print('{}'.format(r))
                            new_pred.append(r)
                        else:
                            pred_random = random.randint(0, 10)
                            new_pred.append(pred_random)

        
        i += 1

    print(len(new_pred))
    #accuracy_score(y_true,y_pred)
    acc = accuracy_score(truelabel,new_pred,normalize=True)
    print("Accuracy: {:.2f}%".format(acc*100))
    # file = open('new_pred3.txt','w')
    # file.write(str(new_pred))
    # file.close()
    plot_confusion(model, new_pred, truelabel,labels=All_list)


All_list = ['IA','IB','MDCK','Para1','Para2','Para3','MK2','CB1','RD']

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
    # predictions = model.predict(X_test)
    # predictions = np.argmax(predictions,axis=1)
    # truelabel = Y_test.argmax(axis=-1)  
    # conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)    
    cm = confusion_matrix(y_true=Y_test,y_pred=X_test)
    plt.figure()
    plot_confusion_matrix(cm, normalize=True,target_names=labels,title='Confusion Matrix')
    plot_confusion_matrix(cm, normalize=False,target_names=labels,title='Non normalize Confusion Matrix')

fusion(X_test=X_test,Y_test=Y_test)