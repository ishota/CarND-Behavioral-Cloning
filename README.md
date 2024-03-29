# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

You can get a self-driving car that runs without departing from the lane.
Please check a details of my solution in [***writeup_report.md***](https://github.com/ishota/CarND-Behavioral-Cloning/blob/master/writeup_report.md).

[gif1]: ./video.gif "video gif"
![alt text][gif1]

## Description
In this project, I learned about deep neural networks and convolutional neural networks to clone driving behavior.
First, we have to collect driving data on cars that have been driven witout departing from the lane.
I used `carnd_p3` data.
Second, I divided the collected data into training data and validation data, and fitted the proposed neural network.
Finaly, I checked driving behavior in driving simulator.

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

A driving behavior is saved in video.
You can create a mp4 video file.

```bash
python video.py run1
```

## Requirement

You can check requirement of anaconda package in environment.yml.

```bash
  - _libgcc_mutex=0.1=main
  - _tflow_1100_select=0.0.1=gpu
  - _tflow_select=2.1.0=gpu
  - absl-py=0.4.1=py35_0
  - asn1crypto=0.24.0=py35_3
  - astor=0.7.1=py35_0
  - blas=1.0=mkl
  - bzip2=1.0.8=h516909a_1
  - ca-certificates=2019.11.27=0
  - certifi=2018.8.24=py35_1
  - cffi=1.11.5=py35h5e8e0c9_1
  - chardet=3.0.4=py35_1
  - click=7.0=py_0
  - cryptography=2.3.1=py35hdffb7b8_0
  - cryptography-vectors=2.3.1=py35_0
  - cudatoolkit=9.2=0
  - cudnn=7.6.4=cuda9.2_0
  - cupti=9.2.148=0
  - cycler=0.10.0=py35hc4d5149_0
  - dbus=1.13.12=h746ee38_0
  - decorator=4.4.1=py_0
  - eventlet=0.23.0=py35_0
  - expat=2.2.6=he6710b0_0
  - ffmpeg=4.2=h167e202_0
  - flask=1.1.1=py_0
  - fontconfig=2.13.0=h9420a91_0
  - freetype=2.9.1=h8a8886c_1
  - gast=0.3.2=py_0
  - glib=2.63.1=h5a9c865_0
  - gmp=6.1.2=hf484d3e_1000
  - gnutls=3.6.5=hd3a4fd2_1002
  - greenlet=0.4.13=py35_0
  - grpcio=1.12.1=py35hdbcaa40_0
  - gst-plugins-base=1.14.0=hbbd80ab_1
  - gstreamer=1.14.0=hb453b48_1
  - h5py=2.8.0=py35h989c5e5_3
  - hdf5=1.10.2=hba1933b_1
  - icu=58.2=h9c2bf20_1
  - idna=2.7=py35_2
  - imageio=2.3.0=py_1
  - intel-openmp=2019.4=243
  - itsdangerous=1.1.0=py_0
  - jinja2=2.10.3=py_0
  - jpeg=9b=h024ee3a_2
  - kiwisolver=1.0.1=py35hf484d3e_0
  - lame=3.100=h14c3975_1001
  - libedit=3.1.20181209=hc058e9b_0
  - libffi=3.2.1=hd88cf55_4
  - libgcc-ng=9.1.0=hdf63c60_0
  - libgfortran-ng=7.3.0=hdf63c60_0
  - libiconv=1.15=h516909a_1005
  - libpng=1.6.37=hbc83047_0
  - libprotobuf=3.6.0=hdbcaa40_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.1.0=h2733197_0
  - libuuid=1.0.3=h1bed415_2
  - libxcb=1.13=h1bed415_1
  - libxml2=2.9.9=hea5a465_1
  - markdown=2.6.11=py35_0
  - markupsafe=1.0=py35h14c3975_1
  - matplotlib=3.0.0=py35h5429711_0
  - mkl=2018.0.3=1
  - mkl_fft=1.0.6=py35h7dd41cf_0
  - mkl_random=1.0.1=py35h4414c95_1
  - moviepy=0.2.3.5=py_0
  - ncurses=6.1=he6710b0_1
  - nettle=3.4.1=h1bed415_1002
  - numpy=1.15.2=py35h1d66e8a_0
  - numpy-base=1.15.2=py35h81de0dd_0
  - olefile=0.46=py_0
  - opencv3=3.1.0=py35_0
  - openh264=1.8.0=hdbcaa40_1000
  - openssl=1.0.2t=h7b6447c_1
  - pandas=0.23.4=py35h04863e7_0
  - pcre=8.43=he6710b0_0
  - pillow=5.2.0=py35heded4f4_0
  - pip=10.0.1=py35_0
  - protobuf=3.6.0=py35hf484d3e_0
  - pycparser=2.19=py_0
  - pyopenssl=18.0.0=py35_0
  - pyparsing=2.4.5=py_0
  - pyqt=5.9.2=py35h05f1152_2
  - pysocks=1.6.8=py35_0
  - python=3.5.6=hc3d631a_0
  - python-dateutil=2.8.1=py_0
  - python-engineio=3.0.0=py_0
  - python-socketio=4.3.0=py_0
  - pytz=2019.3=py_0
  - qt=5.9.6=h8703b6f_2
  - readline=7.0=h7b6447c_5
  - requests=2.19.1=py35_0
  - scikit-learn=0.20.0=py35h4989274_1
  - scipy=1.1.0=py35hfa4b5c9_1
  - setuptools=40.2.0=py35_0
  - sip=4.19.8=py35hf484d3e_0
  - six=1.11.0=py35_1
  - sqlite=3.30.1=h7b6447c_0
  - tbb=2019.8=hfd86e86_0
  - tbb4py=2018.0.5=py35h6bb024c_0
  - tensorboard=1.10.0=py35hf484d3e_0
  - tensorflow=1.10.0=gpu_py35hd9c640d_0
  - tensorflow-base=1.10.0=gpu_py35had579c0_0
  - tensorflow-gpu=1.10.0=hf154084_0
  - termcolor=1.1.0=py35_1
  - tk=8.6.8=hbc83047_0
  - tornado=5.1.1=py35h7b6447c_0
  - tqdm=4.40.0=py_0
  - urllib3=1.23=py35_0
  - werkzeug=0.16.0=py_0
  - wheel=0.31.1=py35_0
  - x264=1!152.20180806=h14c3975_0
  - xz=5.2.4=h14c3975_4
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.3.7=h0b5b093_0

```