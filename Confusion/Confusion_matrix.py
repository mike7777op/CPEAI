import os 
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# tf.enable_eager_execution()
# y_true = [2,1,0,2,2,0,1,1]
# y_pred = [0,1,0,2,2,0,2,1]
# cm = tf.math.confusion_matrix(y_true,y_pred,num_classes = 3).numpy()
class confusion:
# print(cm)

    def plot_confusion_matrix(y_true,y_pred,class_names):
        cm = tf.math.confusion_matrix(y_true,y_pred,num_classes = 3).numpy()
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:,np.newaxis], decimals=2)

        figure = plt.figure(figsize=(8,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        plt.title("Confusion matrix")
            
        tick_index = np.arange(len(class_names))

        plt.yticks(tick_index,class_names)

        plt.xticks(tick_index, class_names, rotation=45)

        plt.colorbar()

        threshold = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > threshold else 'black'
                plt.text(j, i, cm[i,j], horizontalalignment='center', color = color)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.tight_layout()
        plt.show()
        return figure
        

# img = plot_confusion_matrix(cm,[0,1,2])    
# plt.show()

# if  __name__ == "__main__":
#     main()
    
