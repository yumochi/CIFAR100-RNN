'''
Author: Yumo Chi
Modified based on the work from the neural_networks_tutorial.py from pytorch tutorial
as well as codes provided in CS598D's class notes.
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
import h5py
from random import randint
from torch.autograd import Variable

from torchvision import datasets
from torch.utils.data import DataLoader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # kernel
        # 3 input image channel, 32 output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # 32 input image channel, 32 output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # 32 input image channel, 64 output channels, 3x3 square convolution, 1 stride, padding 1 
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
       
        # 32 input image channel, 64 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2)

        # 64 input image channel, 64 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
       
        # 64 input image channel, 64 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv3_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # 64 input image channel, 64 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv3_4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # 64 input image channel, 128 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv4_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # 128 input image channel, 128 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
       
        # 128 input image channel, 128 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv4_3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # 128 input image channel, 128 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv4_4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # 128 input image channel, 256 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv5_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
               
        # 256 input image channel, 256 output channels, 3x3 square convolution, 2 stride, padding 1 
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # batch normalization
        # normalize conv1, conv3, conv5, conv7, conv8 64 output channels
        self.conv1_bn = nn.BatchNorm2d(32)

        #drop out layers for conv1
        self.conv1_dol = nn.Dropout(p=0.1)

        # 2 fully connected layer 
        self.fc1 = nn.Linear(2304, 100)

    def num_flat_features(self, x):
        size = x[0].size() # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # apply dropout relu batch normalization after applying 1st conv
        x = self.conv1_dol(F.relu(self.conv1_bn(self.conv1(x))))
        # apply conv2 block 
        x_1 = x
        x = self.conv2_1(x) 
        x = self.conv2_2(x) 
        x_1 = 
        # apply conv3 block 
        x = self.conv3_1(x) 
        x = self.conv3_2(x) 
        x = self.conv3_3(x) 
        x = self.conv3_4(x) 
        # apply conv4 block 
        x = self.conv4_1(x) 
        x = self.conv4_2(x) 
        x = self.conv4_3(x) 
        x = self.conv4_4(x)
        # apply conv5 block and max pool 2nd of conv in the block
        x = self.conv5_1(x) 
        x = F.max_pool2d(self.conv5_2(x), (2, 2))

        # apply first fully connected layer
        x = self.fc1(x)
        # apply soft_max to the result
        return F.log_softmax(x, dim=1)



def main():
    # set up argparser
    parser = argparse.ArgumentParser(description='hw3')
    # batch-size
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # epochs
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    # learning rate
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    # monte carlo sample times 
    parser.add_argument('--mck', type=int, default=50, metavar='K',
                        help='number of network sampled for monte carlo (default: 50)')

    # gpu setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args = parser.parse_args()

    # test if gpu should be used
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
    # x_train = np.float32(CIFAR10_data['X_train'][:])

    # y_train = np.int32(np.array(CIFAR10_data['Y_train']))

    # # x_test = np.float32(CIFAR10_data['X_test'][:])
    # x_test = np.float32(CIFAR10_data['X_test'][:] )
    # y_test = np.int32( np.array(CIFAR10_data['Y_test']))

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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transformations)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=1)
    batch_size = args.batch_size


    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs
    # L_Y_train = len(y_train)
    model.train()
    # train_loss = []

    for epoch in range(1, args.epochs + 1):
        #Randomly shuffle data every epoch
        # L_Y_train = len(y_train)
        # L_Y_train = 10000
        # I_permutation = np.random.permutation(L_Y_train)

        # x_train = x_train[I_permutation,:]

        # y_train = y_train[I_permutation]
        train_accu = []
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data), Variable(target)

            data, target = data.to(device), target.to(device)
            # x_train_batch = torch.FloatTensor( x_train[i:i+batch_size,:] )
            # y_train_batch = torch.LongTensor( y_train[i:i+batch_size] )
            # data, target = Variable(x_train_batch), Variable(y_train_batch)
            # data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = F.nll_loss(output, target)

            loss.backward()

            # train_loss.append(loss.data[0])

            optimizer.step()
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
               )*100.0
            train_accu.append(accuracy)

        accuracy_epoch = np.mean(train_accu)
        print(epoch, accuracy_epoch)
    
    model.eval()
    test_accu = []
    #L_Y_test = len(y_test)
    # for i in range(0, L_Y_test, batch_size):
        # x_test_batch = torch.FloatTensor( x_test[i:i+batch_size,:] )
        # y_test_batch = torch.LongTensor( y_test[i:i+batch_size] )
        # data, target = Variable(x_test_batch), Variable(y_test_batch)
        # data, target = data.to(device), target.to(device)
    for batch_idx, (data, target) in enumerate(test_loader, 0):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print('test without activating dropout')
    print(accuracy_test)

    k = args.mck
    # perform monte carlo step
    # activate dropout layers
    model.train()
    mc_accu = [[]] * k
    test_accu = []

    for batch_idx, (data, target) in enumerate(test_loader, 0):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        for i in range(k):
            output = model(data)
            if i == 0:   
                mc_prediction = output.data.max(1)[1] # first column has actual prob.
            else:
                mc_prediction += output.data.max(1)[1] # first column has actual prob.

            mc_prediction = torch.div(mc_prediction, i+1)
            mc_accuracy = ( float( mc_prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
            mc_accu[i].append(mc_accuracy)

        output = torch.div(output, k)

        loss = F.nll_loss(output, target)
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(target.data).sum() ) /float(batch_size)
           )*100.0
        test_accu.append(accuracy)

    # print('test with activated dropout')
    # accuracy_test = np.mean(test_accu)
    # print(accuracy_test)

    # print(mc_accu)

    print('showing result with different sample size')
    for i in range(k):
        print('{} samples : {}'. format(batch_size* (i+1), np.mean(mc_accu[i])))

if __name__ == '__main__':
    main()