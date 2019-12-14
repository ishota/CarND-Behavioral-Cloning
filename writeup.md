# **Behavioral Cloning** 

[//]: # (Image References)

[jpg1]: ./picture/center_2016_12_01_13_30_48_287.jpg "test jpg1"
[jpg2]: ./picture/center_2016_12_01_13_33_56_606.jpg "test jpg2"
[gif1]: ./run1.gif "run1 gif"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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
* loss function is mean square error.
* Optimize function is "adam".
* Batch size is 64.

## Result

An output result file is run1.mp4.
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

## Start Guied

You can use the environment.yml to set anaconda environment. 

```bash
conda env create -n tf-gpu -f environment.yml
```

Build and fit the N.N. model. 
I use the data set of `carnd_p3/` prepared by Udacity.

```bash
python model.py
```

Test the N.N. model in Udacity's car simulator.

```bash
python drive.py model.h5 run1
```

A driving behavior is saved in run1.
You can create a video file.

```bash
python video.py run1
```


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
