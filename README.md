# Quntized_Learning_nips2021

## Torch_nn02.py Usage

### Set python Environment
- Use the package on Ubuntu as follows:
	- Check the version of python and following terms 
	- We use python3.8 
### Basic IO

#### Input Files 
- MNIST Data Set 
- CIFAR10 Data Set 

#### output Files 

| Spec | Format | Example|
|---|---|---|
|Neural Network File | torchnn02+ Model name + Algorithm name  + Epoch.pt | torch_nn02ResNetAdam.pt |
| Operation File  | operation + Model name + Algorithm name + Epoch.txt | operation_ResNetAdam100.txt |
| Error Trend File| error_ + Model name + Algorithm name + Epoch.pickle | error_ResNetAdam100.pickle |

### For Cifar-10 Data Set 

#### LeNet
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n LeNet -g 1
~~~

#### ResNet 
~~~
python torch_nn02.py -m Adam -d CIFAR10 -e 100 -n ResNet -g 1 
~~~

#### Note
- When you want not to use CUDA but to use CPU only, don't specify the g option or set the g option as 0.
- For ResNet, due to the size of network, we strongly recommend using the g-option such as "-g 1"

## Quantization Algorithm

For example,  the following setting is for using GPU and initilization at $Q_p = 2$ 

### QSGD

~~~
python torch_nn02.py -m QSGD -d CIFAR10 -e 100 -n ResNet -g 1 -qp 2
~~~

### QtAdamW

~~~
python torch_nn02.py -m QtAdamW -d CIFAR10 -e 100 -n ResNet -g 1 -qp 2
~~~

## Torch_testNN.py Usage
- This program is used for testing the result of learning 
- You should specify the name of network, the network spec. and data set. 
- If there exists an error file in the same folder, it plots the trend of error.
~~~
python torch_testNN.py -d CIFAR10 -n ResNet -ng 1 -e error_ResNetAdam15.pickle -m torch_nn02ResNetAdam.pt 
~~~


## Appendix

#### Torch_nn02.py Help Message

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

