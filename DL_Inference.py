import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
# import torch.utils.data as data
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
from torchvision.transforms import ToTensor
import torchvision.transforms
import torchvision
import os
import numpy as np
import pandas as pd
from datetime import datetime
import sys
# from sklearn import metrics
import random
from PIL import Image
from torch.optim import lr_scheduler
import torchvision.transforms.functional as TF
from torch.autograd import Function
import random
import time
import warnings
warnings.filterwarnings("ignore")#!/usr/bin/env python

# write your unet implementation here
# when using cross entropy loss, you should have output depth of exactly 2
# sigmoid activation must be applied as the last layer in your network to ensure dice metric works properly (also helps cross entropy loss perform better)

# Conv 3*3 in ReLu in paper. A convolutional block for this network is defined as two sequential convolutions with kernal size 3, stride 1, and padding 0. 
class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=0),
            nn.ReLU(inplace=True), # original paper used ReLu, so we specified it here
            nn.Conv2d(out_dim, out_dim, 3, padding=0),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

#For the down side of this network, the number of filters used for the convolutions in each block doubles as you move downwards. (after each pooling layer)
class Down(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            DoubleConv(in_dim, out_dim)
            )

    def forward(self, x):
        return self.down(x)

#For the Up side of the network, the number of filters in each block halves as you move upwards. (after each Up-Convolutional layer)

class Up(nn.Module):
    def __init__(self, in_dim, bilinear = True):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True) # (1, 1024, 28, 28) => (1, 1024, 56, 56)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) #(1, 1024, 56, 56) = > ([1, 1024, 57, 57])
        self.conv1 = nn.Conv2d(in_dim*2, in_dim, kernel_size = 2, stride = 1) #  ([1, 1024, 57, 57]) => ([1, 512, 56, 56])
        self.conv2 = DoubleConv(in_dim*2,  in_dim)  #([1, 512, 56, 56])

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.pad(x1)
        x1 = self.conv1(x1)

        crop = nn.ZeroPad2d(int((x1.size(2) - x2.size(2))/2))
        x2 = crop(x2)  # (1, 64, 512, 512) => (1, 56, 512, 512)

        x = torch.cat([x2, x1], dim = 1)
        x = self.conv2(x)
        return x


#after Up. 
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Make Unet. Doubel Conv first, then Down sampling 4 times. After that, Up sampling 4 times, and then output segmentation map with sigmoid activation. 
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.pre_conv = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(512)
        self.up2 = Up(256)
        self.up3 = Up(128)
        self.up4 = Up(64)
        self.outc = OutConv(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.pre_conv(x) # (1, 1, 572, 572) => (1, 64, 568, 568)
        x2 = self.down1(x1) # (1, 64, 568, 568) => (1, 128, 280, 280)
        x3 = self.down2(x2) # (1, 128, 280, 280) => (1, 256, 136, 136)
        x4 = self.down3(x3) #(1, 256, 136, 136) => (1, 512, 64, 64)
        x5 = self.down4(x4) #  (1, 512, 64, 64) => (1, 1024, 28, 28)
 
        x = self.up1(x5, x4) # (x5: (1, 1024, 28, 28) => ([1, 512, 56, 56]))+ x4((1, 512, 64, 64)) =>
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # => 1, 64, 388, 388
        x = self.outc(x)
        x = self.sig(x)
       
        return x

# dice loss implementation - forward pass can also be used to calculate dice metric
class DiceLoss(Function):
    @staticmethod
    def forward(ctx, inp, target):
        ctx.save_for_backward(inp, target)
        rounded = inp.round()
        inter = torch.dot(rounded.view(-1), target.view(-1)) + .000001
        union = torch.sum(rounded) + torch.sum(target) + .000001

        t = 2.0 * inter.float() / union.float()
        return 1.0 - t

    @staticmethod
    def backward(ctx, grad_output):
        inp, target = ctx.saved_variables
        grad_input = grad_target = None
        rounded = inp.round()

        inter = torch.dot(rounded.view(-1), target.view(-1))
        union = torch.sum(rounded) + torch.sum(target)

        grad_input = -2.0 * grad_output * (.000001 + 2.0 * ((target * union) - (2.0 * rounded * inter))) / (union * union + .000001)

        return grad_input, grad_target

# dataset class
# within root folder there should be a folder with the name partition ('train', 'valid', or 'test').  Within that folder there should be a folder of images called "images" and a folder of corresponding segmentation masks labeled "masks"
# images and masks are assumed to already be 388 x 388
# getitem pads the image so that it will be size 572 x 572, no padding is added to label, so it will remain 388 x 388
class SegmentationDataSet(data.Dataset):
    def __init__(self, root_folder, partition):
        self.data = []
        self.labels = []
        self.partition = partition
        self.pad = nn.ZeroPad2d(92)
        self.tens = ToTensor()

        if not root_folder.endswith('/'):
            root_folder = root_folder + '/'

        img_fol = root_folder + partition + '/images/'
        mask_fol = root_folder + partition + '/masks/'

        for img in os.listdir(img_fol):
            im = np.asarray(Image.open(img_fol + img))
            im = np.uint8(im)
            self.data.append(im)
           
            # use convert mask to 1 channel class labels as required by loss
            mask_im = np.asarray(Image.open(mask_fol + img))
            lbl = np.where(mask_im > 127, 1, 0)
            self.labels.append(lbl)

    def __getitem__(self, index):
        dat = self.data[index]
        lbl = self.labels[index]

        t_dat = self.tens(dat)
        t_dat = self.pad(t_dat)

        t_lbl = torch.tensor(lbl)
        t_lbl = t_lbl.long()
       
        return t_dat, t_lbl

    def __len__(self):
        return len(self.data)

#inference for testing dataset after tuning

# identify where the weights you want to load are
run_id = "DL_training"
weight_fil = run_id + "/best_weights.pth"
print(weight_fil)
# set necessary hyperparameters
batch_size = 1

# load weights
model = torch.load(weight_fil)

# put model in evaluation mode (sets dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results)
model.eval()

# This implementation use CUDA for gpu acceleration
# check if CUDA is available
cuda = torch.cuda.is_available()
if cuda:
    # set the network to use cuda
    model = model.cuda()
    print("CUDA IS AVAILABLE!")
else:
    print("CUDA NOT AVAILABLE!")

# create loaders to feed data to the network in batches
# image size is 500X500, which is larger than 224X224

# testing dataset loader
test_set = SegmentationDataSet('/scratch/dsc381_2019/Homework_5_files/data/', 'test')
testloader = torch.utils.data.DataLoader( dataset = test_set , batch_size= batch_size , shuffle = True)
print('test loader made')


# track metrics over dataset
test_loss = 0.0
epoch_test_loss = 0.0
epoch_test_dice = 0.0
epoch_test_count = 0.0

# loop through eval data
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
#         outputs = outputs[:,1,:,:]
        loss = DiceLoss.apply(outputs.float(), labels.float())

                # track validation loss and dice metric
        epoch_test_loss += loss.item()
        epoch_test_dice += 1.0 - loss.item()
        epoch_test_count += 1.0
           
epoch_test_loss = epoch_test_loss / epoch_test_count
epoch_test_dice = epoch_test_dice / epoch_test_count

epoch_test_dice = [epoch_test_dice]
epoch_test_loss = [epoch_test_loss]


test_result = pd.DataFrame({"test dice": epoch_test_dice, "test loss": epoch_test_loss})
test_result.to_csv(run_id + "/DL_test_result.csv")


# testing dataset loader
train_set = SegmentationDataSet('/scratch/dsc381_2019/Homework_5_files/data/', 'train')
trainloader = torch.utils.data.DataLoader(dataset = train_set , batch_size= batch_size , shuffle = True)
print('train loader made')


# track metrics over dataset
train_loss = 0.0
epoch_train_loss = 0.0
epoch_train_dice = 0.0
epoch_train_count = 0.0

# loop through train data
with torch.no_grad():
    for i, (images, labels) in enumerate(trainloader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
#         outputs = outputs[:,1,:,:]
        loss = DiceLoss.apply(outputs.float(), labels.float())

                # track train loss and dice metric
        epoch_train_loss += loss.item()
        epoch_train_dice += 1.0 - loss.item()
        epoch_train_count += 1.0
           
epoch_train_loss = epoch_train_loss / epoch_train_count
epoch_train_dice = epoch_train_dice / epoch_train_count

epoch_train_dice = [epoch_train_dice]
epoch_train_loss = [epoch_train_loss]


train_result = pd.DataFrame({"train dice": epoch_train_dice, "train loss": epoch_train_loss})

train_result.to_csv(run_id + "/train_result.csv")



# valid dataset loader
valid_set = SegmentationDataSet('/scratch/dsc381_2019/Homework_5_files/data/', 'valid')
validloader = torch.utils.data.DataLoader( dataset = valid_set , batch_size= batch_size , shuffle = True)
print('valid loader made')


# track metrics over dataset
valid_loss = 0.0
epoch_valid_loss = 0.0
epoch_valid_dice = 0.0
epoch_valid_count = 0.0

# loop through valid data
with torch.no_grad():
    for i, (images, labels) in enumerate(validloader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
#         outputs = outputs[:,1,:,:]
        loss = DiceLoss.apply(outputs.float(), labels.float())

                # track train loss and dice metric
        epoch_valid_loss += loss.item()
        epoch_valid_dice += 1.0 - loss.item()
        epoch_valid_count += 1.0
           
epoch_valid_loss = epoch_valid_loss / epoch_valid_count
epoch_valid_dice = epoch_valid_dice / epoch_valid_count

epoch_valid_dice = [epoch_valid_dice]
epoch_valid_loss = [epoch_valid_loss]


valid_result = pd.DataFrame({"valid dice": epoch_valid_dice, "valid loss": epoch_valid_loss})

valid_result.to_csv(run_id + "/valid_result.csv")



