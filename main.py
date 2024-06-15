# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:53:34 2024

@author: IDAR
"""

import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader
import statistics
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import random
from model import AUFA
from tqdm import tqdm
from itertools import zip_longest
from pre_model import Transformerpre
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import dataset

trs = []
tes = []

Correct_nums=[]
test_accs=[]
test_AUCs=[]
RECs=[]
PREs=[]
F1s=[]
ROOT_PATH = '../sample_data/' 

modelroot='train'

#Conducting multiple training sessions
for m in range(5):  
    # Setting different random seeds
    torch.manual_seed(m)
    np.random.seed(m)
    random.seed(m)
    os.makedirs(os.path.join('re', modelroot, str(m)), exist_ok=True)
    
    
    def get_device():
      return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    def augmentation_loss( out1, out2, sr_epsilon=0.6, sr_loss_p=0.5, args=None):
        
        prob1_t = F.softmax(out1, dim=1)
        prob2_t = F.softmax(out2, dim=1)
    
        prob1 = F.softmax(out1, dim=1)
        log_prob1 = F.log_softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        log_prob2 = F.log_softmax(out2, dim=1)
        
        if random.random() <= sr_loss_p:
            
              log_prob2 = F.log_softmax(out2, dim=1)
              mask1 = (prob1_t.max(-1)[0] > sr_epsilon).float() #Boolean mask
              aug_loss = ((prob1 * (log_prob1 - log_prob2)).sum(-1) * mask1).sum() / (mask1.sum() + 1e-6)

              
        else:
            
              log_prob1 = F.log_softmax(out1, dim=1)
              mask2 = (prob2_t.max(-1)[0] > sr_epsilon).float()
              aug_loss = ((prob2 * (log_prob2 - log_prob1)).sum(-1) * mask2).sum() / (mask2.sum()+1e-6)
             
        return aug_loss

    #source domain data
    full_dataset  =dataset.Fmri()
    
    #target domain data
    full_dataset2  =dataset.Fmri2()
    
    loader1 = DataLoader(dataset=full_dataset, num_workers=0, batch_size=32,shuffle=True)
    loader2 = DataLoader(dataset=full_dataset2, num_workers=0, batch_size=32,shuffle=True)
    
    
    # Setting hyperparameters
    device=get_device()
    print(f'DEVICE: {device}')
    learning_rate=0.00001
    num_epoch =45
    
    
    # Introduction to pre-training
    premodel = Transformerpre()
    premodel.load_state_dict(torch.load(os.path.join('result', 'model', 'modelpre.pth')))
    premodel_dict = premodel.state_dict()
    
    #Initialize training parameters
    model = AUFA()
    model_dict = model.state_dict()
    premodel_dict = {k: v for k, v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #training
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0
    

        model.train()
        for i, (data, data2) in enumerate(zip(loader1, loader2)):

            x, y = data
            x=x.to(torch.float32)
            y=y.to(torch.float32)
            x=x.to(device)
            y=y.to(device)
            x2, y2 = data2
            x2=x2.to(torch.float32)
            x2=x2.to(device)
            inputs=torch.cat((x,x2),dim=0)
            optimizer.zero_grad()
            outputs, mmd_loss=model(x,x2)
            
            #cross-entropy loss
            classification_loss = criterion(outputs.narrow(0, 0, y.size(0)), y.long())
            
            #Augmentation-based optimization loss
            A_loss = 0
            outputs_tgt = outputs.narrow(0, y.size(0), inputs.size(0)-y.size(0))
            outputs_tgt_perturb = outputs.narrow(0, inputs.size(0),
                                              inputs.size(0) - y.size(0))
            A_loss = augmentation_loss(outputs_tgt, outputs_tgt_perturb, sr_epsilon=0.6,
                                         sr_loss_p=0.5)
            
            batch_loss = classification_loss + A_loss  *1 + mmd_loss * 1 
            
            _, train_pred = torch.max(outputs.narrow(0, 0, y.size(0)), 1)
            
            batch_loss.backward()
            optimizer.step()
            train_acc += (train_pred.cpu() == y.cpu()).sum().item()
            train_loss += batch_loss.item()
            
        if epoch==num_epoch-1:

            torch.save(model.state_dict(), os.path.join('re', modelroot, str(m), 'model.pth'))  

        
        #test
        model.eval()
        
        test_lossduo = 0
        test_correct = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0
        
        te_auc_y_gt = []
        te_auc_y_pred = []
        
        with torch.no_grad():
            for i, data in enumerate(loader2):
                x, y = data  
                x = x.to(torch.float32)
                y = y.to(torch.float32)
                x = x.to(device)
                y = y.to(device)
                outputs, mmd_loss = model(x,x2)
                batch_loss = criterion(outputs, y.long())
                test_pred = outputs.argmax(1)
                prod = outputs.softmax(1)
                test_acc += (
                        test_pred.cpu() == y.cpu()).sum().item()
                test_loss += batch_loss.item()
                
                test_lossduo += batch_loss.item()
                test_correct += sum(test_pred.cpu().numpy() == y.cpu().numpy())
                TN_tmp, FP_tmp, FN_tmp, TP_tmp = confusion_matrix(y.cpu().numpy(), test_pred.cpu().numpy(), labels=[0, 1]).ravel()
                TN += TN_tmp
                FP += FP_tmp
                FN += FN_tmp
                TP += TP_tmp
                te_auc_y_gt.extend(y.cpu().numpy())
                te_auc_y_pred.extend(prod[0:,1].cpu().numpy())
                
            test_lossduo /= len(loader2)

            REC = TP / (TP + FN)  # recall
            PRE = TP / (TP + FP)  # Precision
            F1=2*(PRE*REC)/(PRE+REC) #f1 score
            test_accduo = (TP+TN) / (TP + FP + TN + FN)
            test_AUC = roc_auc_score(te_auc_y_gt, te_auc_y_pred)
    
    
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(full_dataset), train_loss / len(loader1),
                test_acc / len(full_dataset2), test_loss / len(loader2)
            ))
            
        
    Correct_num=test_correct
    test_accduo=test_accduo
    test_AUC=test_AUC
    REC=REC
    PRE=PRE
    Correct_nums.append(test_acc)
    test_accs.append(test_accduo) 
    test_AUCs.append(test_AUC)
    RECs.append(REC)
    PREs.append(PRE)
    F1s.append(F1)
    
     
    tr=train_acc / len(full_dataset)
    te=test_acc / len(full_dataset2)
    trs.append(tr)
    tes.append(te)
    print('[{:03d}] Train Acc: {:3.6f}'.format(m,tr))
    print('[{:03d}] Test Acc: {:3.6f}'.format(m,te))

  
average_result_Correct_num = np.mean(Correct_nums)
average_result_test_acc = np.mean(test_accs)
average_result_test_AUC = np.mean(test_AUCs)
average_result_REC = np.mean(RECs)
average_result_PRE = np.mean(PREs)
average_result_F1 = np.mean(F1s)
acc_std = np.std(test_accs,ddof=1)
auc_std = np.std(test_AUCs,ddof=1)
rec_std = np.std(RECs,ddof=1)
pre_std = np.std(PREs,ddof=1) 
f1_std = np.std(F1s,ddof=1)
print("Average Result Correct_num:", average_result_Correct_num)
print("Average Result test_acc:", average_result_test_acc) 
print("Average Result test_AUC:", average_result_test_AUC)
print("Average Result recall:", average_result_REC) 
print("Average Result precision:", average_result_PRE) 
print("Average Result F1:", average_result_F1) 
print("std_acc:",acc_std)
print("std_auc:",auc_std)
print("std_rec:",rec_std)
print("std_pre:",pre_std)
print("std_f1:",f1_std)
      
