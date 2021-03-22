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
model = load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/GAP_vs_GMP/Multi/0124_AVG_IPE_Multi_model_1_acc0.9808.h5')

X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/New_data_Generator/test_npy/Y_test.npy')




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
                return 12
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


def fusion(X_test,Y_test):

    pred1 = model.predict(X_test)[0]
    pred2 = model.predict(X_test)[1]
    pred3 = model.predict(X_test)[2]
    predictions1 = np.argmax(pred1,axis=1)
    predictions2 = np.argmax(pred2,axis=1)
    predictions3 = np.argmax(pred3,axis=1)
    
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
        #a,b,c位置
        a = predictions1[i]
        b = predictions2[i]
        c = predictions3[i]

        if a == 0:
            if b == 0:
                if c == 0:
                    # print('a0 vs b0 vs c0')
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
                        l=[0,3,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

                elif c == 1:
                    # print('a0 vs b0 vs c1')
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
                        l=[0,3,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a0 vs b0')                   
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a0')
                        new_pred.append(0)
                    elif ans == 2:
                        print('b0')
                        new_pred.append(3)
                    else:
                        l=[0,3]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 1:
                if c == 0:
                    # print('a0 vs b1 vs c0')
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
                        l=[0,4,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a0 vs b1 vs c1')
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
                        l=[0,4,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a0 vs b1')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a0')
                        new_pred.append(0)
                    elif ans == 2:
                        print('b1')
                        new_pred.append(4)
                    else:
                        l=[0,4]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 2:
                if c == 0:
                    # print('a0 vs b2 vs c0')
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
                        l=[0,5,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a0 vs b2 vs c1')
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
                        l=[0,5,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a0 vs b2')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a0')
                        new_pred.append(0)
                    elif ans == 2:
                        print('b2')
                        new_pred.append(5)
                    else:
                        l=[0,5]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    # print('a0 vs b3 vs c0')
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
                        l=[0,6,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a0 vs b3 vs c1')
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
                        l=[0,6,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a0 vs b3')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a0')
                        new_pred.append(0)
                    elif ans == 2:
                        print('b3')
                        new_pred.append(6)
                    else:
                        l=[0,6]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 4:
                if c == 0:
                    # print('a0 vs c0')
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
                    # print('a0 vs c1')
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
                    print('a0')
                    new_pred.append(0)


        elif a == 1:
            if b == 0:
                if c == 0:
                    # print('a1 vs b0 vs c0')
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
                        l=[1,3,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a1 vs b0 vs c1')
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
                        l=[1,3,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a1 vs b0')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a1')
                        new_pred.append(1)
                    elif ans == 2:
                        print('b0')
                        new_pred.append(3)
                    else:
                        l=[1,3]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 1:
                if c == 0:
                    # print('a1 vs b1 vs c0')
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
                        l=[1,4,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a1 vs b1 vs c1')
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
                        l=[1,4,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a1 vs b1')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a1')
                        new_pred.append(1)
                    elif ans == 2:
                        print('b1')
                        new_pred.append(4)
                    else:
                        l=[1,4]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 2:
                if c == 0:
                    # print('a1 vs b2 vs c0')
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
                        l=[1,5,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a1 vs b2 vs c1')
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
                        l=[1,5,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a1 vs b2')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a1')
                        new_pred.append(1)
                    elif ans == 2:
                        print('b2')
                        new_pred.append(5)
                    else:
                        l=[1,5]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    # print('a1 vs b3 vs c0')
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
                        l=[1,6,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a1 vs b3 vs c1')
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
                        l=[1,6,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a1 vs b3')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a1')
                        new_pred.append(1)
                    elif ans == 2:
                        print('b3')
                        new_pred.append(6)
                    else:
                        l=[1,6]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 4:
                if c == 0:
                    # print('a1 vs c0')
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
                    # print('a1 vs c1')
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
                    print('a1')
                    new_pred.append(1)

        elif  a == 2:
            if b == 0:
                if c == 0:
                    # print('a2 vs b0 vs c0')
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
                        l=[2,3,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a2 vs b0 vs c1')
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
                        l=[2,3,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a2 vs b0')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a2')
                        new_pred.append(2)
                    elif ans == 2:
                        print('b0')
                        new_pred.append(3)
                    else:
                        l=[2,3]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 1:
                if c == 0:
                    # print('a2 vs b1 vs c0')
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
                        l=[2,4,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a2 vs b1 vs c1')
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
                        l=[2,4,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a2 vs b1')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a2')
                        new_pred.append(2)
                    elif ans == 2:
                        print('b1')
                        new_pred.append(4)
                    else:
                        l=[2,4]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 2:
                if c == 0:
                    # print('a2 vs b2 vs c0')
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
                        l=[2,5,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a2 vs b2 vs c1')
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
                        l=[2,5,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a2 vs b2')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a2')
                        new_pred.append(2)
                    elif ans == 2:
                        print('b2')
                        new_pred.append(5)
                    else:
                        l=[2,5]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
            
            elif b == 3:
                if c == 0:
                    # print('a2 vs b3 vs c0')
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
                        l=[2,6,7]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 1:
                    # print('a2 vs b3 vs c1')
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
                        l=[2,6,8]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                elif c == 2:
                    # print('a2 vs b3')
                    ans = score2(p1,p2)
                    if ans == 1:
                        print('a2')
                        new_pred.append(2)
                    elif ans == 2:
                        print('b3')
                        new_pred.append(6)
                    else:
                        l=[2,6]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)

            elif b == 4:
                if c == 0:
                    # print('a2 vs c0')
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
                    # print('a2 vs c1')
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
                    print('a2')
                    new_pred.append(2)
        
        elif a == 3:
            if b == 0:
                if c == 0:
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
                    # print('b0 vs c1')
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
                    print('b0')
                    new_pred.append(3)

            elif b == 1:
                if c == 0:
                    # print('b1 vs c0')
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
                    # print('b1 vs c1')
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
                    print('b1')
                    new_pred.append(4)

            elif b == 2:
                if c == 0:
                    # print('b2 vs c0')
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
                    # print('b2 vs c1')
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
                    print('b2')
                    new_pred.append(5)
            
            elif b == 3:
                if c == 0:
                    # print('b3 vs c0')
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
                    # print('b3 vs c1')
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

                    # new_pred.append('b3 vs c1')
                elif c == 2:
                    print('b3')
                    new_pred.append(6)

            elif b == 4:
                if c == 0:
                    print('c0')
                    new_pred.append(7)
                elif c == 1:
                    print('c1')
                    new_pred.append(8)
                elif c == 2:
                    print('None')
                    ans = score1(p1,p2,p3)
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
                        l = [0,1]
                        r = choice(l)
                        print('{}'.format(r))
                        new_pred.append(r)
                    else:
                        pred_random = random.randint(0, 8)
                        new_pred.append(pred_random)

        
        i += 1

    print(len(new_pred))
    #accuracy_score(y_true,y_pred)
    acc = accuracy_score(truelabel,new_pred,normalize=True)
    print("Accuracy: {:.2f}%".format(acc*100))
    file = open('new_pred3.txt','w')
    file.write(str(new_pred))
    file.close()

fusion(X_test=X_test,Y_test=Y_test)