from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
    # rotation_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

gener = datagen.flow_from_directory(
    r'/home/pmcn/workspace/CPE_AI/Resnet50/tool/Data_augumentation/ImageDataGenerator/generator',
    target_size=(1038,1388),
    batch_size=300,
    shuffle=False,
    save_to_dir=r'/home/pmcn/workspace/CPE_AI/Resnet50/tool/Data_augumentation/ImageDataGenerator/generator',
    save_prefix='trans_',
    save_format='jpg'
)
i=0
for batch in gener:
    i += 1
    if i >3:
        break