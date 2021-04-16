import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time
import os
import copy
from torch.optim import lr_scheduler 
import torch.hub
from torch.optim import lr_scheduler
from torch.nn import Dropout,Dropout2d,Dropout3d
import matplotlib.pyplot as plt
import torch.nn.functional as F


K.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Resize(250),
#         transforms.CenterCrop(250),
#         transforms.RandomHorizontalFlip(p=1),
#         transforms.RandomVerticalFlip(p=1),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(250),
#         transforms.CenterCrop(250),
#         #transforms.RandomHorizontalFlip(p=1),
#         #transforms.RandomVerticalFlip(p=1),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }


X_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/X_train.npy')
Y_train = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/train_npy/Y_train.npy')
X_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/X_test.npy')
Y_test = np.load('/home/pmcn/workspace/CPE_AI/Resnet50/Balance_data/500/Data_generator/Single/test_npy/Y_test.npy')


print('X_train shape : ',X_train.shape)
print('Y_train shape : ',Y_train.shape)
print('X_test shape : ',X_test.shape)
print('Y_test shape : ',Y_test.shape)


#將我們先前用 numpy 建立了資料轉成 tensor

X_np = torch.from_numpy(X_train)
Y_np = torch.from_numpy(Y_train)
# data_dir = 'C:\\Users\\user\\Desktop\\ia'
# image_datasets = {
#     x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
#     for x in ['train', 'val']
#     }
# dataloaders = {
#     x: torch.utils.data.DataLoader(image_datasets[x], batch_size=60,  shuffle=True, num_workers=4,pin_memory=True)
#     for x in ['train', 'val']
#     }

# dataset_sizes = {
#     x: len(image_datasets[x])
#     for x in ['train', 'val']
#     }
#print("Classes: ")
class_names = image_datasets['train'].classes
#print(image_datasets['train'].classes)
#device = torch.device("cpu")
# 图片展示的函数

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    
                                   
    
      
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    i=0
    #losses1=[]
    #losses2=[]
    #ep1=[]
    #ep2=[]
    losses1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    losses2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ep1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ep2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #aaaa, bbbb = torch.max(outputs, 1)
                    #print(aaaa)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                      #直接呼叫 backward() 就能幫我們計算所有的梯度了
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                
            if phase == 'train':
                scheduler.step()
            if phase == 'train':
                
               epoch_loss = running_loss / dataset_sizes[phase]
               epoch_acc = running_corrects.double() / dataset_sizes[phase]
               epoch_acc1 = running_corrects / dataset_sizes[phase]
               acc1 = epoch_acc1.cpu().numpy().item()
               #print(acc1)
               print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
               
            if phase == 'val':
               epoch_loss1 = running_loss / dataset_sizes[phase]
               epoch_acc1 = running_corrects.double() / dataset_sizes[phase]
               epoch_acc2 = running_corrects / dataset_sizes[phase]
               acc2 = epoch_acc2.cpu().data.numpy().item()
               #print(acc2)
               #aa=losses1.append(epoch_loss)
               #aa1=losses2.append(epoch_loss1)
               losses1[i]=epoch_loss
               ep1[i]= acc1
               aa=losses1
               aa1=ep1 
               #print(aa1)
               #bb=ep1.append(epoch_acc)
               #bb1=ep2.append(epoch_acc1)
               losses2[i]=epoch_loss1
               ep2[i]= acc2
               bb=losses2
               bb1=ep2
               #print(bb1)
               i=i+1
               print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss1, epoch_acc1))
               
           
            if phase == 'val' and epoch_acc1 > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best-model-parameters.pt')
            
        print()
         
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    
    #np.save('my_array-1', aa)
    #np.save('my_array', aa1)
    #np.save('my_array1', bb)
    #np.save('my_array2', bb1)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
 
 class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #resnet = torch.hub.load(
           #'moskomule/senet.pytorch',
           #'se_resnet50',
           # pretrained=True,
        #)
        resnet = torchvision.models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        model_34=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*model_34)
        self.fc=nn.Linear(2048, 9)
    def forward(self, x ):
        batch_size, C, H, W = x.size()
        c_in = x.view(batch_size, C, H, W)
        #####################以下把圖片(32，3，300,300) 輸入進入resnet50########### 
        out = self.resnet(c_in)
        out = out.view(out.size(0),-1)
        out = out.view(out.size(0),-1)
        print(out.shape)
        out = self.fc(out)
        
        return out
       

        

 modelpicture = CNN()
 #torch.save(modelpicture, 'ia2.pt')
 model_ft = modelpicture.to(device)
 
 criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean')

 # Observe that all parameters are being optimized
 
 optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001,  weight_decay=0)
 #optimizer_ft =optim.Adagrad(model_ft.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
 #optimizer_ft=optim.RMSprop(model_ft.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
 #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
 #optimizer_ft=torch.optim.Adadelta(model_ft.parameters(), lr=0.1, rho=0.9, eps=1e-06, weight_decay=0)
 #optimizer_ft = torch.optim.Adamax(model_ft.parameters(),  lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
 


  #以上6組優化器 

 # Decay LR by a factor of 0.1 every 7 epochs
 exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  
 model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)
 
 
 
 

 