'''
Author: Yumo Chi
Modified based on the work from the neural_networks_tutorial.py from pytorch tutorial
as well as work from yunjey's RNN tutorial, especially regarding treatment of 
shortcuting and use of filter block
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse

import torchvision
import torchvision.transforms as transforms
# import torchsample as ts

import numpy as np
from random import randint
from torch.autograd import Variable

from torchvision import datasets
from torch.utils.data import DataLoader

# model_url for selection on different pretrained nets

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# function to add pre trained net
def resnet18(pretrained = True) :
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained :
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir ='./'))
    return model

# 2 filter block
class ResNet_Block(nn.Module):
    ''' constructor function
        param:
            in_channels: input channels
            out_channels: output channels
            stride: stride for filters in the block
            downsample: downsample function 
    '''
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNet_Block, self).__init__()
        # input image channel, output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
        # batch normalization
        self.conv_1_bn = nn.BatchNorm2d(out_channels)

        # input image channel, output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # batch normalization
        self.conv_2_bn = nn.BatchNorm2d(out_channels)

        # downsample flag
        self.downsample = downsample

    # forward function
    def forward(self, x):
        shortcut_input = x
        # F.relu(self.conv1_bn(self.conv1(x)))
        out = F.relu(self.conv_1_bn(self.conv_1(x)))
        out = self.conv_2_bn(self.conv_2(out))

        if self.downsample:
            shortcut_input = self.downsample(shortcut_input)

        out += shortcut_input
        return out

# ResNet - Residual Network
class ResNet(nn.Module):
    ''' constructor function
        param:
            block: blocks for the Residual Network
            lays: list to specify number of blocks in each layer
            num_classes: number of classes 
    '''
    def __init__(self, block, layers, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 32
        # kernel
        # 3 input image channel, 32 output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # batch normalization
        # normalize conv 32 output channels
        self.conv1_bn = nn.BatchNorm2d(32)

        #drop out layers for conv
        self.conv1_dol = nn.Dropout(p=0.1)

        # conv2 block, 2 3x3 conv, 32 input, 32 output, 1 stride
        self.bl_2 = self.make_block_layer(block, 32, 1, layers[0])
        # conv2 block, 4 3x3 conv, 32 input, 64 output, 2 stride
        self.bl_3 = self.make_block_layer(block, 64, 2, layers[1])
        # conv2 block, 4 3x3 conv, 64 input, 128 output, 2 stride
        self.bl_4 = self.make_block_layer(block, 128, 2, layers[2])
        # conv2 block, 2 3x3 conv, 128 input, 256 output, 2 stride
        self.bl_5 = self.make_block_layer(block, 256, 2, layers[3])

        # 1 fully connected layer 
        self.fc1 = nn.Linear(1024, 100)

    # function to create layer based on number of blocks 
    def make_block_layer(self, block, out_channels, stride=1, blocks=2):
        downsample = None
        # decide if downsample is needed
        if (stride != 1) or (self.in_channels != out_channels):
            # use conv2d to subsample if needed
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        filters = []
        filters.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            filters.append(block(out_channels, out_channels))
        # connect the filters 
        return nn.Sequential(*filters)
    # function to help reduce input to a 1D tensor 
    def num_flat_features(self, x):
        size = x[0].size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # apply dropout relu batch normalization after applying 1st conv
        x = self.conv1_dol(F.relu(self.conv1_bn(self.conv1(x))))
        # apply block layer 2
        x = self.bl_2(x)
        # apply block layer 3 
        x = self.bl_3(x) 
        # apply block layer 4
        x = self.bl_4(x)
        # apply max_pool and block layer 5 
        x = F.max_pool2d(self.bl_5(x), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        # apply first fully connected layer
        x = self.fc1(x)
        # apply soft_max to the result
        return F.log_softmax(x, dim=1)


def main():
    # set up argparser
    parser = argparse.ArgumentParser(description='hw3')
    # batch-size
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    # epochs
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    # learning rate
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    # gpu setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()

    # test if gpu should be used
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # adding in data augmentation transformations
    train_transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # just transform to tensor for test_data
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # data loader
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1)
    batch_size = args.batch_size


    # initialize the residual net
    # model = ResNet(ResNet_Block,[2,4,4,2]).to(device)
    model = resnet18()
    model.fc = nn.Linear(512, 100)
    model = model.cuda()

    # # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    model.train()
    printFlag = True
    for epoch in range(1, args.epochs + 1):
        #Randomly shuffle data every epoch
        train_accu = []
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            # upsample for resnet 18
            data = F.interpolate(data, size=(224, 224))
            data, target = Variable(data), Variable(target)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # change loss function
            CEL = nn.CrossEntropyLoss().cuda()
            loss = CEL(output, target)

            # start backprop
            loss.backward()
            # optimize
            optimizer.step()

            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
               )*100.0
            train_accu.append(accuracy)

        accuracy_epoch = np.mean(train_accu)
        print(epoch, accuracy_epoch)
    
    model.eval()
    test_accu = []
    for batch_idx, (data, target) in enumerate(test_loader, 0):
        # upsample for resnet 18
        data_upsample = F.interpolate(data, size=(224, 224))
        data_upsample, target = Variable(data_upsample), Variable(target)
        data_upsample, target = data_upsample.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_upsample)

         # change loss function
        CEL = nn.CrossEntropyLoss().cuda()
        loss = CEL(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print(accuracy_test)


if __name__ == '__main__':
    main()