# ScatNet-Demo
Network-enabled efficient image restoration for 3D microscopy of turbid biological specimens

ScatNet is based on CSBDeep (https://github.com/CSBDeep/CSBDeep) and adjusted for 3D image restoration of turbid biological specimens.

# Requirements
ScatNet is built with Python, Tensorflow and Keras. Technically there are no limits to the operation system to run the code, but Windows system is recommonded, on which the software has been tested.

The inference process of the ScatNet can run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the inference is highly recommended.

The inference process has been tested with:

Windows 10 pro (version 1903)
Python 3.6.7 (64 bit)
tensorflow 1.15.0
Intel Core i7-5930K CPU @3.50GHz
Nvidia GeForce RTX 2080 Ti
