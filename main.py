import json
import csv
import os
import os.path
import numpy as np
from utils.load_signals_student import PrepDataStudent
from utils.load_signals_teacher import PrepDataTeacher
from utils.prep_data_student import train_val_test_split_student
from utils.prep_data_teacher import train_val_test_split_teacher
from models.student import ConvLSTM_student
from models.teacher import ConvLSTM_teacher
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

def main():
    torch.cuda.empty_cache()
    
    #hyperparameters
    T = 4
    alpha = 0.6
    beta = 0.5    
    epochs=25

    #Prepare settings and storage
    with open('teacher_settings.json') as f:
        teacher_settings = json.load(f)
    with open('student_settings.json') as k:
        student_settings = json.load(k)
    makedirs(str(teacher_settings['cachedir']))
    makedirs(str(teacher_settings['resultdir']))
    makedirs(str(student_settings['cachedir']))
    makedirs(str(student_settings['resultdir']))
    teacher_results = []
    student_results = []

    #Patients to be tested on, patient 9 may cause memory crashes
    targets = ['1','2','3','5','9', '10','13','18','19','20','21','23']
  
    for target in targets:
         ###We begin by training the teacher on the entire target###
        
        #Data preparation, we need to store the preprocessed seizure files for classification
        ictal_X, ictal_y = PrepDataTeacher(target, type='ictal', settings=teacher_settings).apply()
        
        interictal_X, interictal_y = PrepDataTeacher(target, type='interictal', settings=teacher_settings).apply()
        #The second patient has only two seizures, so we use 50/50 split, otherwise 0.35 split
        if target in '2': X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_teacher(ictal_X, ictal_y, interictal_X, interictal_y, 0.5, 0.5)
        else: X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_teacher(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)   
        teacher = ConvLSTM_teacher(X_train.shape).to('cuda')

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader_teacher  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        
        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=32)


        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4)
        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            teacher.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader_teacher:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()
                outputs = teacher(X_batch)
                loss = ce_loss(F.softmax(outputs, dim=1), Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)

        pbar.close()   
        teacher.eval()
        teacher.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = teacher(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        teacher_results.append(auc_test)
    
        ###This is where we train the student, do not update the teacher weights!!!###
        
        ictal_X, ictal_y = PrepDataStudent(target, type='ictal', settings=student_settings).apply()
        interictal_X, interictal_y = PrepDataStudent(target, type='interictal', settings=student_settings).apply()
        if target in '2': X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_student(ictal_X, ictal_y, interictal_X, interictal_y, 0.5, 0.5)
        else: X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_student(ictal_X, ictal_y, interictal_X, interictal_y, 0.25, 0.35)   
        student = ConvLSTM_student(X_train.shape).to('cuda')

        Y_train       = torch.tensor(y_train).type(torch.LongTensor).to('cuda')
        X_train       = torch.tensor(X_train).type(torch.FloatTensor).to('cuda')
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader  = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        
        if X_val is not None and y_val is not None:
            y_val       = torch.tensor(y_val).type(torch.LongTensor)
            X_val       = torch.tensor(X_val).type(torch.FloatTensor)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader  = DataLoader(dataset=val_dataset, batch_size=32)

        ce_loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(student.parameters(), lr=5e-4)

        pbar = tqdm(total=epochs)
        for epoch in range(epochs):
            student.train()
            teacher.eval()
            total_loss = 0
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to("cuda"), Y_batch.to("cuda")
                optimizer.zero_grad()

                #Below is the key aspect of KDTT, the distillation process
                with torch.no_grad():
                   #Forward pass on the teacher
                   teacher_logits = teacher(X_batch)
                student_logits = student(X_batch)
                soft_targets = F.softmax(teacher_logits / T, dim=1)
                soft_prob = F.log_softmax(student_logits / T, dim=1)
                soft_targets_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T**2) 
                #Uncomment below to switch between KL Loss or MSE loss, MSE loss does not use temperature
                #soft_targets_loss = F.mse_loss(student_logits, teacher_logits)
                label_loss = ce_loss(F.softmax(student_logits, dim=1), Y_batch)
                #To train on the standard student without distillation, uncomment below, and replace the loss
                #loss = label_loss
                loss = alpha * soft_targets_loss + beta*label_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            pbar.update(1)

        pbar.close()   
        student.eval()
        student.to('cuda')
        X_tensor = torch.tensor(X_test).float().to('cuda')
        y_tensor = torch.tensor(y_test).long().to('cuda')

        with torch.no_grad():
            predictions = student(X_tensor)

        predictions = predictions[:, 1].cpu().numpy()
        auc_test = roc_auc_score(y_tensor.cpu(), predictions)
        print('Test AUC is:', auc_test)
        student_results.append(auc_test)

    teacher_avg_auc = sum(teacher_results) / len(teacher_results)
    student_avg_auc = sum(student_results) / len(student_results)

    return student_avg_auc



if __name__ == '__main__':
    main()