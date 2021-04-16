from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from keras.optimizers import Adam,SGD,sgd
from keras.models import load_model
from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize


nb_classes = 9
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/MK2_less/Single/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/MK2_less/Single/test_npy/Y_test.npy')



model = load_model('/home/pmcn/workspace/CPE_AI/Resnet50/checkpoint3/Balance_train/500/LR:test/Single_lr:0.00003_1/Single_model_lr:0.00003_1.h5')



Y_pred = model.predict(X_test)
predictions = np.argmax(pred,axis=1)
truelabel = Y_test.argmax(axis=-1) 

# Binarize the output
truelabel = label_binarize(truelabel, classes=[i for i in range(nb_classes)])
predictions = label_binarize(predictions, classes=[i for i in range(nb_classes)])

# Y_pred = [np.argmax(y) for y in Y_pred]  
# Y_test = [np.argmax(y) for y in Y_test]

# Binarize the output
# Y_test = label_binarize(Y_test, classes=[i for i in range(nb_classes)])
# Y_pred = label_binarize(Y_pred, classes=[i for i in range(nb_classes)])


# micro：多分类　　
# weighted：不均衡数量的类来说，计算二分类metrics的平均
# macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
# precision = precision_score(Y_test, Y_pred, average='micro')
# recall = recall_score(Y_test, Y_pred, average='micro')
# f1_score = f1_score(Y_test, Y_pred, average='micro')
# accuracy_score = accuracy_score(Y_test, Y_pred)
# print("Precision_score:",precision)
# print("Recall_score:",recall)
# print("F1_score:",f1_score)
# print("Accuracy_score:",accuracy_score)

precision = precision_score(truelabel, predictions,average='micro')
recall = recall_score(truelabel, predictions, average='micro')
f1_score = f1_score(truelabel, predictions, average='micro')

print('Precision score:',precision)
print('Recall score:',recall)
print('F1_score:',f1_score)

print(classification_report(truelabel, predictions ,target_names=All_list1,digits=4))


# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
# 横坐标：假正率（False Positive Rate , FPR）


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(truelabel[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(truelabel.ravel(), predictions.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])



# Plot all ROC curves
lw = 2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
# plt.savefig("../images/ROC/ROC_5分类.png")
plt.show()


print("--- %s seconds ---" % (time.time() - start_time))
