# Honda Sales Price Prediction w/ Neural Network

The project was completed as a part of the Neural Network course at Eszterhazy Karoly Catholic University during my Computer Science BSc in 2023

## Project Description
This project implements a simple Neural Network utilizing Pandas and Numpy. The NN performs data analytics, cleaning, processing, and model building. It supports training and testing processes with configurable parameters such as activation functions, learning rate, and epochs.

### Key features
* Supports Sigmoid and Tanh activation functions
* Weights are initialized randomly within a range
* Backpropagation for error correction and updating weights
* Training continues until MSE reaches a threshold or a specified number of epochs is completed
* Utilizes data normalization and one-hot encoding

## Components
### dataset
This package contains the original and modified datasets of the Honda sales data. 

### src
The NN ops dir utilizes data_processor.py to clean and prepare the data with classic normalization and one-hot encoding.

### neu2
An alternate approach for running the dataset.

## Results
The dataset was run on all general vehicle categories sold by Honda at the time. To decrease the likelyhood of outliers, car ages were limited to only a few years. With car prices reaching values in the 30+ thousand dollar range, the model reached an average error of $500 compared to the actual sales price.
