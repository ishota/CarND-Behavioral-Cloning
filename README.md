# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, I learned about deep neural networks and convolutional neural networks to clone driving behavior.

[gif1]: ./run1.gif "run1 gif"
![alt text][gif1]

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
You can create a mp4 video file.

```bash
python video.py run1
```
