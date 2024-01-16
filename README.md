# Knowledge Distillation Through Time For Future Event Prediction #

This repository contains code from the following paper: Skye Gunasekaran, Jason K. Eshraghian, Ruomin Zhu, Zdenka Kuncic "Knowledge Distillation Through Time For Future Event Prediction". {insert link} {insert date}.

## Requirements ## 

Inidividual module requirements for the project can be found within the file "requirements.txt". The version of Python used was 3.11.7. The verson of CUDA used was 12.1 on Windows 11.

## Code Execution ##

The programs main code, including parameters and the training loop, can be found in "Main.py". Before running the code, ensure that the CHBMIT dataset has been downloaded and placed within the "Dataset" folder. It can be found at this link: https://physionet.org/content/chbmit/1.0.0/. There are three different experiments that can be performed, namely, KL Divergence, Mean Square Error, and the Baseline student. These can be decided using the argument parser "--mode={experiment type}', where {experiment type} can have the values of "KL", "MSE", or "baseline". 

## Parameter Tuning ## 

As mentioned in our paper, we use key three parameters in KDTT for the distillation process, alpha, beta, and temperature. 
* Alpha: scales the cross-entroy loss
* Beta: scales the distillation loss
* Temperature: scales the softmax for KL loss. 

## Preprocessing Details ##

Finer details of the project, including the preprocessing code, can be found within the "utils" folder. This folder contains the preprocessing for both the teacher and student models. In summary, we first classify the data as either preictal, ictal or interical. Then, we take short time fourier transforms at a sampling rate of 256hz, and convert this into a numpy array. For the student model, we use a seizure occurence period of 30 minutes, and a seizure prediction horizon of 5 minutes. Our models are Convolutional LSTMs, and can be found within the folder "models".