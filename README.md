# CIFAR100-ResNet

Just a simple residual neural net that runs on CIFAR100. It showcases some simple concepts like loading pretrained network, in this case resnet 18 and how to incorporate short cuts to create resnet on pytorch. The code is inspired by yunjey's pytorch tutorial, specifically the use of layer blocks, see https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py#L76-L113 as well as work from CS598 D from UIUC, which is a Deep Learning course I am taking.

The code is divided into two parts, one is a simple residual net from scratch, ie main.py. The other is utiliized a pretained network, pretrained_net.py


## Getting Started

To get started on the project is very easy, just

```
git clone git@github.com:yumochi/CIFAR100-ResNet.git
```

### Prerequisites

To run the code you will need the following:

# Python

(Refer to https://www.python.org/downloads/)

# Pytorch 

```
# Python 3.x
pip3 install torch torchvision
```

```
# Python 2.x`
pip install torch torchvision
```

(Refer to https://pytorch.org/get-started/locally/ for more info.)



# Torchvision

```
pip install torchvision
```

(Refer to https://pypi.org/project/torchvision/0.1.8/ )

# h5py (Not necessary)

```
pip install h5py
```

h5py was originally used to import image data, but the code is adopted to use Torchvision

Comment out code if not needed

## Running the tests

# To run the code, just run

```
python main.py/pre_trained.py
```
# Set parameters with argparser

For a list of terminal commands for the argparser, refer to texts below or check in main.py for all parameters

# Set batch-size with --batch-size x

x has to be an integer

```
python main.py --batch-size 16
```

# Set epoch number with --epochs x

x has to be an integer

```
python main.py --epochs 30
```

# Set learning rate with --lr x

x has to be an float

```
python main.py --lr 0.0001
```

# Set sample number in monte carlo approximation with --mck x

x has to be an integer

```
python main.py --mck 16
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Yumo Chi** - *Initial work* - [CIFAR-100-ResNet](https://github.com/yumochi/CIFAR100-ResNet)


## Acknowledgments

* Pytorch Developers
* UIUC CS598D's professor and tas.
=======
