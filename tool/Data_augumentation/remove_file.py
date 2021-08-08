import os 

path = os.listdir('/home/pmcn/workspace/Test_Code/ImageDataGenerator/MDCK')
file_end = r'.xml'
i = '.xml'
for file in path:
    if os.path.splitext(file)[1] == '.xml': 
        os.remove(r'/home/pmcn/workspace/Test_Code/ImageDataGenerator/MDCK'+ '/' + file)