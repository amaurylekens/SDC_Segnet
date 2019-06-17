# Encoder-Decoder for road Segmentation

This repository contains the files necessary to realize the semantic segmentation of an image to identify the shape of a road. The project uses the Udacity autonomous car simulator.

The algorithm used is an encoder-decoder neuron network.

## Folders

```bash
├── segnet.py
├── train.py
├── labelizer.py
├── test 
│   ├── test.py
│   └── result.png
├── prepare_label.py
├── live_segmentation.py
├── README.md
└── .gitignore
```

* segnet.py : a class which contain the encoder-decoder structure with a train and a predict method
* train.py : fit the weight of the network and store it
* labelizer.py : function which help to create the label from a set of image
* prepare_label.py : functions which prepare data and label for the training
* live_segmentation.py : road segmentation in live with the udacity simulator

## Run

### Train

1. Run the train.py file

### Live segmentation

1. Run the Udacity simulator and go in the self-driving mode
2. Run the drive.py file

## Dependencies

* opencv
* tensorflow
* keras

## Explications

### Segmentation

The purpose of the segmentation is to assign the pixels belonging to the route to one class and the other pixels to another class.

![segnet](https://github.com/amaurylekens/SDC_Segnet/blob/master/images/segnet.png)

The network input is an rgb image and the output is an image where each pixel is assigned to a class.

### Labelization


