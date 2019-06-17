# Encoder-Decoder for road Segmentation

This repository contains the files necessary to realize the semantic segmentation of an image to identify the shape of a road. The project uses the Udacity autonomous car simulator.

## Folders

```bash
├── segnet.py
├── train.py
├── labelizer.py
├── test 
│   ├── test.py
│   └── result.png
├── prepare_label.py
├── compute_output_img.py
├── live_segmentation.py
├── model_weight.hdf5
├── README.md
└── .gitignore
```

* segnet.py : a class which contain the encoder-decoder structure with a train and a predict method
* train.py : fit the weight of the network and store it
* labelizer.py : function which help to create the label from a set of image
* prepare_label.py : functions which prepare data and label for the training
* compute_output_img.py : function which transform the output of the network in segmented image
* live_segmentation.py : road segmentation in live with the udacity simulator
* test.py : test the model on new image
* model_weight.hdf5 : store the weight of the trained model

## Run

### Train

1. Prepare a folder with images captured with the Udacity simulator
2. Run the train.py file

### Live segmentation

1. Run the Udacity simulator and go in the self-driving mode
2. Run the live_segmentation.py file
3. Drives the car (there is no self-driving mode for the moment)

## Dependencies

* keras, tensorflow, scikit-learn, openCV
* pandas, numpy, matplotlib, base64, io
* socketio, eventlet

## Algorithme

### Segmentation

The purpose of the segmentation is to assign the pixels belonging to the route to one class and the other pixels to another class.

![segnet](https://github.com/amaurylekens/SDC_Segnet/blob/master/images/segnet.png)

The algorithm used is an encoder-decoder neuron network. The input of the network is an rgb image and the output is an image where each pixel is assigned to a class. 

The encoder is composed of several layers of convolutions, normalizations and pooling. The decoder has an inverse architecture and the pooling layers are replaced by upsampling layers.

The goal is to use this architecture in combination with a CNN to predict the direction of the car in the Udacity simulator.

### Labelization

To train the model, one needs a dataset containing images with their labeled version. To achieve this, we use the labelizer.py function, this function uses image processing techniques to label an image. The main steps are:

![segnet](https://github.com/amaurylekens/SDC_Segnet/blob/master/images/labelization.png)

* Perform two edge detections: a sensitive and a less sensitive
* Select the upper part of the sensitive and lower detection of the less sensitive and paste them
* Find the first white dots on the left and right from the middle line
* Find the best regressions for each edge from the find points

### Result

Training on 80 segmented images and with 7 epochs : 

<p align="center">
  <img src="https://github.com/amaurylekens/SDC_Segnet/blob/master/test/result.png"/>
</p>

## References

* https://medium.com/coinmonks/semantic-segmentation-deep-learning-for-autonomous-driving-simulation-part-1-271cd611eed3
* https://arxiv.org/pdf/1511.00561.pdf
