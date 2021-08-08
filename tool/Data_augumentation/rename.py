import os
import random

path = '/home/pmcn/workspace/Test_Code/ImageDataGenerator/ADV/'
files = os.listdir(path)
print('files')

n=0
for i in files:
    newname = str(n+1)+'.jpg'
    os.chdir(path)
    os.rename(i,newname)
    print(i+'>>>'+newname)
    n=n+1