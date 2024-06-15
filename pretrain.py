# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 10:57:59 2023

@author: IDAR
"""
import os
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


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
ROOT_PATH = './sample_data/' 


#Data used for pre-training
class Fmri(object):
     def read_data(self):
         
         data = scipy.io.loadmat(ROOT_PATH + 'source.mat')
         bold =data['DATA']
         A =bold[0]
         pc_l=[]
         for i in range(len(A)):
             pc= np.corrcoef(A[i].T)
             pc = np.nan_to_num(pc)
             pc_l.append(pc)
         X =np.array(pc_l)
         y = np.squeeze(data['lab'])

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

full_dataset  =Fmri()

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(full_dataset,lengths=[train_size,test_size])

train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_set,num_workers=0, batch_size=32,shuffle=True)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)


        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):

        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)


        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn    
    
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs):   

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        n_layers=2  

        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):       
        enc_outputs=enc_inputs
        enc_self_attns = []
        for layer in self.layers:
            #
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.dense_1 =nn.Linear( 13456,4096)
        self.dense_2 =nn.Linear(4096,2)
        self.dense_3 =nn.Linear(256,2)
        #self.relu =nn.ReLU()
        self.dropout =nn.Dropout(p=0.5)

    def forward(self, enc_inputs):  

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        x = enc_outputs.view(enc_outputs.size(0), -1)
        source_1 =self.dense_1(x)
        x =self.dense_2(source_1)
        
        return x


    
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'


os.makedirs(os.path.join('result', 'model'), exist_ok=True)
os.makedirs(os.path.join('result', 'summary'), exist_ok=True)
#SAVE_PATH = '../result/pretrain/'
device=get_device()
print(f'DEVICE: {device}')

# Setting hyperparameters
learning_rate=0.001
num_epoch = 20
src_len = 116
tgt_len = 116
d_model = 116
d_ff = 116
d_k = d_v = 32
n_heads = 4
model = Transformer().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    test_acc = 0.0
    test_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        x, y = data
        x=x.to(torch.float32)
        y=y.to(torch.float32)
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output=model(x)
        batch_loss = criterion(output, y.long())
        _, train_pred = torch.max(output, 1)  # get the index of the class with the highest probability
        # print(train_pred)
        batch_loss.backward()
        optimizer.step()
        train_acc += (train_pred.cpu() == y.cpu()).sum().item()
        train_loss += batch_loss.item()
        
    if epoch ==5:
        print('yes')
        #Save pre-training parameters
        torch.save(model.state_dict(), os.path.join('result', 'model', 'modelpre.pth'))


    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data  
            x = x.to(torch.float32)
            y = y.to(torch.float32)
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            batch_loss = criterion(output, y.long())
            _, test_pred = torch.max(output, 1)
            test_acc += (
                    test_pred.cpu() == y.cpu()).sum().item()
            test_loss += batch_loss.item()


        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Test Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
            test_acc / len(test_set), test_loss / len(test_loader)
        ))