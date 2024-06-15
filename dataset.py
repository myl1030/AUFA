# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:36:01 2024

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

ROOT_PATH = './sample_data/'

#Source Domain Data
class Fmri(object):
     def read_data(self):
         
         site20 = scipy.io.loadmat(ROOT_PATH + 'source.mat')
         bold =site20['DATA']
         A =bold[0]
         pc_l=[]
         for i in range(len(A)):
             pc= np.corrcoef(A[i].T)
             pc = np.nan_to_num(pc)
             pc_l.append(pc)
         X =np.array(pc_l)
         y = np.squeeze(site20['lab'])
         return X,y

     def __init__(self):
         super(Fmri,self).__init__()
         X,y =self.read_data()
         self.X =torch.from_numpy(X)
         self.y =torch.from_numpy(y)
         self.n_samples =X.shape[0]
         
     def __getitem__(self, index):
          return self.X[index],self.y[index]
      
     def __len__(self):
         return self.n_samples




#target Domain Data
class Fmri2(object):
     def read_data(self):
 
         site20 = scipy.io.loadmat(ROOT_PATH + 'target.mat')
         bold =site20['DATA']
         A =bold[0]
         pc_l=[]
         
         for i in range(len(A)):
             pc= np.corrcoef(A[i].T)
             pc = np.nan_to_num(pc)
             pc_l.append(pc)
         X =np.array(pc_l)
         y = np.squeeze(site20['lab'])

         return X,y

     def __init__(self):
         super(Fmri2,self).__init__()
         X,y =self.read_data()
         self.X =torch.from_numpy(X)
         self.y =torch.from_numpy(y)
         self.n_samples =X.shape[0]
         
     def __getitem__(self, index):
          return self.X[index],self.y[index]

     def __len__(self):
         return self.n_samples
     
        
     
        
     
        
     
        