from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Resnet50_package import New_Resnet50
from keras.applications import keras_modules_injection


@keras_modules_injection
def Resnet50_2(*args, **kwargs):
    return New_Resnet50.ResNet50(*args, **kwargs)


@keras_modules_injection
def decode_predictions(*args, **kwargs):
    return New_Resnet50.decode_predictions(*args, **kwargs)


@keras_modules_injection
def preprocess_input(*args, **kwargs):
    return New_Resnet50.preprocess_input(*args, **kwargs)