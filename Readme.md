# Space Efficient TREC for Enabling Deep Learning on Microcontrollers


Space efficient TREC (denoted as TREC in the following) is a new form of convolution optimized for microcontrollers. It makes *trainsient redundancy* detection and avoidance an inherent part of the CNN architecture, and the determination of the best configurations for redundancy elimination part of CNN backward propagation.

TREC is currently implemented as a new lightweight high-level API of Pytorch for defining, training and evaluating complex models. This directory contains code for training and evaluating several compact Convolutional Neural Networks (CNNs) using TREC.

It contains scripts that will allow you to train models from scratch and evaluate models on both server and Microcontrollers(MCUs). 

## Table of contents

  
<a  href="#Setup">Experimental setup</a><br>

<a  href="#Install">Installation</a><br>

<a  href='#Train'>Training from scratch</a><br>

<a  href='#Pretrain'>Using pre-trained models</a><br>

<a  href='#MCU'>Evaluating on MCUs</a><br>

## Experimental setup
<a  id='Setup'></a>

 - Training Machine:  
	 - An NVIDIA GeForce RTX A6000 GPU server with 20-core 3.60GHz Intel Core i7-12700K processor, 128GB of RAM, and 48GB of GPU memory.
	 - Pytorch-1.10.1 (open-source software with a BSD license) .
 - Deployment Machine:
	 - An STM32F469NI MCU with 324KB SRAM and 2MB Flash.
	 - An STM32F746ZG MCU with 320KB SRAM and 1MB Flash.
	 - CMSIS-NN kernel optimized for Arm Cortex-M devices.


## Installation (on Training Machine)
<a  id='Install'></a>

In this section, we describe the steps required to install the appropriate prerequisite packages.
TREC requires Python version 3.6 or later.

### Install PyTorch
Since we implement TREC as an extension of PyTorch, the prerequisite is the installation of PyTorch. Pytorch is available at its [official website](https://pytorch.org/get-started/locally/).  Select your preferences and run the install command.
TREC requires torch version 1.3.0 or later.

### Install the TREC package
To install the TREC package, simply run:
```
python setup.py install
```

After a few minutes, TREC will exist in the environment as a PyTorch extension package named *trec*.
Now we can use TREC by importing both *torch* and *trec* packages in Python.


## Training a model from scratch (on Training Machine)
<a  id='Train'></a>

We provide an easy way to train a model from scratch using Cifar-10 dataset. The following example demonstrates how to train SqueezeNet using the default parameters.

```shell
TRAIN_DIR=/tmp/TREC/examples/EXP
DATASET_DIR=/tmp/TREC/data
python train_model.py \
--checkpoint_path=${TRAIN_DIR} \
--dataset_path=${DATASET_DIR} \
--model_name=SqueezeNet
```
For simplicity, we put the scripts for training in the ``examples/scrips/`` directory. You can start training by simply executing the following command.

 1. ``cd examples/scrips/ ``
 2. ``bash train_squeeze_on_cifar10_template.sh``


## Pre-trained models
<a  id='Pretrain'></a>

Because training models from scratch can be a very computationally intensive process requiring multiple hours, we provide various pre-trained models under directory `examples/pre_trained_models/`.


## Evaluating on MCUs (on Deployment Machine)
<a  id='MCU'></a>

We provide the evaluation code for TREC.

The first step in deploying the trained keyword spotting models on microcontrollers is quantization, which is described [here](https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Quant_guide.md). This directory consists of example codes and steps for running a quantized DNN model on any Cortex-M board using [mbed-cli](https://github.com/ARMmbed/mbed-cli) and [CMSIS-NN](https://github.com/ARM-software/CMSIS_5) library. It also consists of an example of integration of the TREC model onto a Cortex-M development board to demonstrate real time inference on live streaming data.

### Get the CMSIS-NN library and install mbed-cli

Clone [CMSIS-5](https://github.com/ARM-software/CMSIS_5) library, which consists of the optimized neural network kernels for Cortex-M.

```shell
cd MCU_eval
git clone https://github.com/ARM-software/CMSIS_5.git
```

Install [mbed-cli](https://github.com/ARMmbed/mbed-cli) and its python dependencies:

```shell
pip install mbed-cli
```

There can be updates for the CMSIS_5 library which cause conflicts, and the library will provide instructions on how to resolve these file dependancies issues (if applicable).

### Build and run

We refer to the website [CMSIS-nn image recognition](https://developer.arm.com/documentation/102689/0100/Build-basic-camera-application) for a general workflow for running NN models on microcontrollers (unfortunately, this website is not available at this time).

In the `trecFunctions` directory, move `arm_convolve_HWC_q7_RGB_reuse.c`  and `arm_convolve_HWC_q7_LCNN.c` under the directory `CMSIS_5/CMSIS/NN/Source/ConvolutionFunctions`. Then move `arm_nnfunctions.h` under the directory `CMSIS_5/CMSIS/NN/Include` replacing the existing one. Generally, the function `arm_convolve_HWC_q7_RGB_cluster` will be used for trec, while `arm_convolve_HWC_q7_basic` and `arm_convolve_HWC_q7_RGB` are used for original networks. So replace `arm_convolve_HWC_q7_basic` or `arm_convolve_HWC_q7_RGB` for `arm_convolve_HWC_q7_RGB_cluster`, and we can have the trec version of the network.

We have our image data loaded to the `camera_with_nn.cpp` file, so inference is run on this input data. First create a new project and install any python dependencies prompted when project is created for the first time after the installation of mbed-cli.

```shell
mbed new trec --mbedlib 
```



Fetch the required mbed libraries for compilation.

```shell
cd trec
mbed deploy
```

Since there can be version discrepancies for CMSIS-NN library, we include our code in `CMSISNN_Webinar.zip ` in this archive. So an alternative is to get the code running on MCU is simply by downloading the `CMSISNN_Webinar.zip ` file.


Now you can have MCU board (in our case, it's DISCO_F469NI) connected to your computer. Then, compile and run the code for the mbed board. The inference time will show up on the screen of the MCU board.

```shell
mbed compile -t GCC_ARM -m DISCO_F469NI --source . --source ../squeeze_complex_bypass --source ../CMSIS_5/CMSIS/NN/Include --source ../CMSIS_5/CMSIS/NN/Source --source ../CMSIS_5/CMSIS/Core/Include --source ../CMSIS_5/CMSIS/DSP/Include --source ../CMSIS_5/CMSIS/DSP/Source --source ../CMSIS_5/CMSIS/DSP/PrivateInclude -j8   --flash --sterm
```

We use a timer `t` in the code to record time.
For other networks, change the source files during compilation time.
One thing to note is that we provide the original squeezeNet, users can easily select some convolutional layer and replace it with the trec version and the results shown is under the situation where all the convolutional layers are replaced with trec.