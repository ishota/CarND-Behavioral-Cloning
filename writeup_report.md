# **Behavioral Cloning** 

[//]: # (Image References)

[jpg1]: ./picture/center_2016_12_01_13_30_48_287.jpg "test jpg1"
[jpg2]: ./picture/center_2016_12_01_13_33_56_606.jpg "test jpg2"
[gif1]: ./video.gif "video gif"

## Writeup

This is a project for Udacity lesson of Self-driving car engineer.
I created a self-driving car that is able to drive araound the lake course by behavioral cloning.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Collected data

I used a carnd_p3 behavior data prepared by Udacity.
In the behavior data, a steering angle for the image showing the front center, front right and front left of the car is recorded.
The steering angle that circulate without departing from the course are recorded in two directions including reverse rotation.
These following two images were actually used for model fitting.

![alt text][jpg1]
![alt text][jpg2]

## Preprocess data

I preprocessed data for learning successfuly.

* RGB to YUV conversion: Learning not only by color but also by luminance and color difference, aiming at increasing data and generalization of data
* Invert image and steering: To increase training data when turning.
* Blur an image with GaussianBlur: To increase generalization performance.
* Probably not adopted when steer is small: Many images with a steer value of 0 in which caused over-fitting.

## Proposed model

I used N.N. model for cloning behaviror of data set.
I showed a detail of used model below.

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda (Lambda)              (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d (Cropping2D)      (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d (Conv2D)              (None, 43, 158, 24)       1824      
_________________________________________________________________
dropout (Dropout)            (None, 43, 158, 24)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 77, 36)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 37, 48)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 33, 64)         76864     
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten (Flatten)            (None, 8448)              0         
_________________________________________________________________
dense (Dense)                (None, 100)               844900    
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 11        
=================================================================
Total params: 994043
Trainable params: 994043
Non-trainable params: 0
_________________________________________________________________
```

The first lambda layer outputs a normalized images to prevent the so-colled vanishing gradients.
The second layer outputs an image cropped from the top and bottom of the image.
From the therd layer to the sixth layer, there are convolutional layers to capture small features in an image.
In these convolutional layers, I used dropout technich to suppress over fitting.
I put a flatten layer to connect convolutional layer to all connected layer.
Finaly, The model output a steering angle.

### Training parameters

* Input image shape is 160 x 320 x 3.
* Divided into training data and validation data at a ratio of 8 to 2.
* loss function is mean square error.
* Optimize function is "adam".
* Batch size is 64.

## Result

An output result file is video.mp4.
I put here a gif file that shows the results.
You can find that no tire leave the drivable portion of the trach surface.

![alt text][gif1]

And here I show a history of learning.
It can be seen that the loss values ​​of training data and validation data are gradually decreasing and converged.

```python
Epoch 1/20  loss: 0.0213 - val_loss: 0.0193
Epoch 2/20  loss: 0.0153 - val_loss: 0.0169
Epoch 3/20  loss: 0.0157 - val_loss: 0.0172
Epoch 4/20  loss: 0.0138 - val_loss: 0.0153
Epoch 5/20  loss: 0.0143 - val_loss: 0.0145
Epoch 6/20  loss: 0.0136 - val_loss: 0.0169
Epoch 7/20  loss: 0.0134 - val_loss: 0.0141
Epoch 8/20  loss: 0.0136 - val_loss: 0.0157
Epoch 9/20  loss: 0.0132 - val_loss: 0.0156
Epoch 10/20 loss: 0.0131 - val_loss: 0.0141
Epoch 11/20 loss: 0.0125 - val_loss: 0.0155
Epoch 12/20 loss: 0.0117 - val_loss: 0.0149
Epoch 13/20 loss: 0.0126 - val_loss: 0.0144
Epoch 14/20 loss: 0.0118 - val_loss: 0.0137
Epoch 15/20 loss: 0.0111 - val_loss: 0.0143
Epoch 16/20 loss: 0.0116 - val_loss: 0.0152
Epoch 17/20 loss: 0.0114 - val_loss: 0.0147
Epoch 18/20 loss: 0.0107 - val_loss: 0.0145
Epoch 19/20 loss: 0.0101 - val_loss: 0.0157
Epoch 20/20 loss: 0.0101 - val_loss: 0.0142
```