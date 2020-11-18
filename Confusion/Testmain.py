from Confusion_matrix import confusion
import os 
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()
y_true = [2,1,0,2,2,0,1,1]
y_pred = [0,1,0,2,2,0,2,1]
class_name = [0,1,2]
cm = tf.math.confusion_matrix(y_true,y_pred,num_classes = 3).numpy()
c = confusion()
y = c.plot_confusion_matrix(y_true,y_pred)
