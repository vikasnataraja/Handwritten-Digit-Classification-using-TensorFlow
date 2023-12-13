## Handwritten-Digit-Classification-using-TensorFlow

This repo is a simple python script that teaches beginners about deep learning. I used this Jupyter notebook/python script to teach a class last year about the awesomeness of CNNs. It builds a simple convolutional neural network to classify handwritten digits from the MNIST dataset. Originally created by [@Aymeric Damien](https://github.com/aymericdamien), modified and updated with plots and descriptions.

## Install the packages

It is recommended to create a new conda environment for this tutorial. Execute the following commands in a terminal.

```
conda create -n cnn python=3.10 -y
conda activate cnn
conda install -c conda-forge numpy matplotlib jupyter -y
conda install tensorflow==2.12.0 keras==2.12.0 -y
```

NOTE: Anaconda installation of tensorflow tends to be slow (30 - 60 minutes). This is an active issue that is out of our control (for now). Please feel free to run this in the background beforehand.