# Learning Algorithm with Monotonically Increasing Resolution of Quantization

This repository is the official implementation of "Learning Algorithm with Monotonically Increasing Resolution of Quantization"

## Requirements

We write the python test codes based on the PyTorch.
If the PyTorch is already installed on your system, pass this stage. If not, recomend the following URL :
~~~
https://pytorch.kr/get-started/locally/
~~~

To install requirements: pytorch 
~~~
pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f 
~~~

### System 
We have tested the codes on a Linux system.  The detailed specification of the system is represented in the following Table.

| item |  Specification |
|---|---|
| OS     | Linux, Ubuntu 16.0.4  |
| CUDA   | CUDA version 10.2     | 
| GPU    | GeForce GTX 1080Ti    |
| PyTorch| 1.8.2 Stable     | 
| Python | 3.6 ~ 3.8.8      |   

## Training
To train the proposed model in the paper, run this command:
~~~
python torch_nn02.py -d CIFAR10 -e 20 -n ResNet -g 1 -qp 2 -l 0.25 -m QSGD
~~~
The parameters described in the above command is as follows:

| Parameter | meaning |
|---|---|
| -d CIFAR10 | The name of data set is "CIFAR10"  |
| -e 20      | The number of trotal epoch is "20" | 
| -n ResNet  | The model of neural network is "ResNet" |
| -g 1       | Use GPU |
| -qp 2      | Initial index for qunatization. It means 2^{-2} |
| -l 0.25    | learning rate is 0.25. we recommend 0.00390625~0.03125 |
| -m QSGD    | The proposed learning algorithm. The other proposed algorithm is QSGD | 

If you want to use a conventional learning such as SGD or Adam to comapare the proposed algorithm, use following command. 
~~~
python torch_nn02.py -d CIFAR10 -e 20 -n ResNet -g 1 -l 0.25 -m Adam
~~~

If you want to aware the detailed usage of the provided code, Please read the "Usage of the provided code" and "Torch_nn02.py Help Message" in Appendix

## Evaluation
Use Torch_testNN.py 
- This program is used for testing the result of learning 
- You should specify the name of network, the network spec. and data set. 
- If there exists an error file in the same folder, it plots the trend of error.

To evaluate my model on ImageNet, run:
~~~
python torch_testNN.py -d CIFAR10 -n ResNet -ng 1 -e error_ResNetAdam15.pickle -m torch_nn02ResNetAdam.pt 
~~~
- Illustration of the parameters

|  Parameter | meaning |
|---|---|
| -d CIFAR10 | The name of data set is "CIFAR10"  |
| -n ResNet  | The model of neural network is "ResNet" |
| -ng 1      | No plotting of error tresnd, if you want plotting of error trend, deglect the parameter or set 0| 
| -e error_ResNetAdam15.pickle | set the error trend file |
| -m torch_nn02ResNetAdam.pt   | set the trained model    |  

## Pre-trained Models
You can download pretrained models here:


## Results
Our model achieves the following performance on :



## Appendix
### Usage of the provided code
#### Basic IO
##### Input Files 
- MNIST Data Set 
- CIFAR10 Data Set 

##### output Files 

| Spec | Format | Example|
|---|---|---|
|Neural Network File | torchnn02+ Model name + Algorithm name  + Epoch.pt | torch_nn02ResNetAdam.pt |
| Operation File  | operation + Model name + Algorithm name + Epoch.txt | operation_ResNetAdam100.txt |
| Error Trend File| error_ + Model name + Algorithm name + Epoch.pickle | error_ResNetAdam100.pickle |

#### Data Sets
##### LeNet
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n LeNet -g 1
~~~

##### ResNet 
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n ResNet -g 1 
~~~

#### Note
- When you want not to use CUDA but to use CPU only, don't specify the g option or set the g option as 0.
- For ResNet, due to the size of network, we strongly recommend using the g-option such as "-g 1"

### Quantization Algorithm
For example,  the following setting is for using GPU and initilization at $Q_p = 2$ 

#### QSGD
~~~
python torch_nn02.py -m QSGD -d CIFAR10 -e 100 -n ResNet -g 1 -qp 2
~~~

#### QtAdamW
~~~
python torch_nn02.py -m QtAdamW -d CIFAR10 -e 100 -n ResNet -g 1 -qp 2
~~~


### Torch_nn02.py Help Message

~~~
usage: torch_nn02.py [-h] [-g DEVICE] [-l LEARNING_RATE] [-e TRAINING_EPOCHS] [-b BATCH_SIZE] [-f MODEL_FILE_NAME] [-m MODEL_NAME] [-n NET_NAME] [-d DATA_SET] [-a AUTOPROC]

====================================================
torch_nn02.py : Based on torch module
                    Written by Jinwuk @ 2021-03-10
====================================================
Example : python torch_nn02.py

optional arguments:
  -h, --help            show this help message and exit
  -g DEVICE, --device DEVICE
                        Using [0]CPU or [1]GPU
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning_rate
  -e TRAINING_EPOCHS, --training_epochs TRAINING_EPOCHS
                        training_epochs
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch_size
  -f MODEL_FILE_NAME, --model_file_name MODEL_FILE_NAME
                        model file name
  -m MODEL_NAME, --model_name MODEL_NAME
                        model name 'SGD', 'Adam', 'Manual'
  -n NET_NAME, --net_name NET_NAME
                        Network name 'CNN', 'LeNet', 'ResNet'
  -d DATA_SET, --data_set DATA_SET
                        data set 'MNIST', 'CIFAR10'
  -a AUTOPROC, --autoproc AUTOPROC
                        Automatic Process without any plotting
~~~

###  Torch_testNN.py Help Message

~~~
>python torch_testNN.py -h
usage: test pytorch_inference [-h] [-g DEVICE] [-d DATASET] [-m MODELFILE]
                              [-n MODELCLASS] [-e ERROR_TREND_FILE]
                              [-ng NO_GRAPH] [-p PLOTTING_POINTS]

====================================================
torch_testNN.py : Based on torch module
                    Written by Jinwuk @ 2021-03-12
====================================================
Example : python torch_testNN.py

optional arguments:
  -h, --help            show this help message and exit
  -g DEVICE, --device DEVICE
                        Using [0]CPU or [1]GPU
  -d DATASET, --dataset DATASET
                        Name of Data SET 'MNIST', 'CIFAR10'
  -m MODELFILE, --modelfile MODELFILE
                        Name of Model file
  -n MODELCLASS, --modelclass MODELCLASS
                        Name of Model class 'CNN', 'LeNet', 'ResNet'
  -e ERROR_TREND_FILE, --error_trend_file ERROR_TREND_FILE
                        Error Trend File such as 'error_file.pickle'
  -ng NO_GRAPH, --no_graph NO_GRAPH
                        Plot the erroe trend [1] of learning or not[0 :
                        default]
  -p PLOTTING_POINTS, --plotting_points PLOTTING_POINTS
                        plotting_points for Error Trend
  -rl NUM_RESNET_LAYERS, --num_resnet_layers NUM_RESNET_LAYERS
                        The number of layers in a block to ResNet                        
~~~

