Knowledge Distillation Through Time (KDTT) For Future Event Prediction

This github repository will allow a user to perform KDTT on the CHBMIT seizure dataset. 

The folder "models" describes both the teacher and student model, which are convolutional LSTMS. The only difference between these two models is that the teacher performs max pooling, wheras the student performs average pooling. 

The folder "utils" is incredibly important, and contains all of the preprocessing and data loading for this project. Within the "utils" folder, we have the sinal processing for the student and teacher, the data splitting, as well as some additionall helper files. As mentioned in the paper, we have a seizure occurence period of 30 minutes, and a seizure prediction horizon of 5 minutes, and this can be found within the singal processing files. These files depend on a sampling csv file which will be located in the main repository of the project. 

The folder "dataset" will contain the CHBMIT dataset. Additionally, we have placed some necessary files within the folder which will be called on during the preprocessing stage.

Main.py is the executable for the program, and contains the training for both the student and teacher, as well as the hyperparameters. For a brief explaination, the parameter alpha determines the scaling of the the cross entropy, beta determines with the scaling of the distillation loss, temperature affects the Kullback Leibler loss function, and epochs are the number of training cycles for each patient. 