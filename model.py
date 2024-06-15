# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:52:52 2024

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

#Setting hyperparameters
src_len = 116
tgt_len = 116
d_model = 116
d_ff = 116 
d_k = d_v = 32 
n_heads = 4 


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

    def forward(self, enc_inputs,per=True): 
        
        B = enc_inputs.shape[0]
        y=enc_inputs
        enc_self_attns = []

        if self.training:
            perturb_layer = random.choice([0,1])
            
        for layer, blk in enumerate(self.layers):

            if self.training:
                if per:
                    if layer ==perturb_layer:
                        idx = torch.flip(torch.arange(B // 2, B), dims=[0])
                        ym = y[B // 2:] + (y[idx]-y[B // 2:]).detach() * 0.3
                        y = torch.cat((y, ym))

                    y, enc_self_attn = blk(y)
                    enc_self_attns.append(enc_self_attn)

                else:
                    y, enc_self_attn = blk(y)
                
            else:
                y, enc_self_attn = blk(y)

        return y, enc_self_attns
  
class AUFA(nn.Module):
    def __init__(self):
        super(AUFA, self).__init__()
        self.encoder = Encoder()
        self.dense_1 =nn.Linear( 13456,4096)
        self.dense_2 =nn.Linear(4096,2)
        self.dense_3 =nn.Linear(256,2)
        #self.relu =nn.ReLU()
        self.dropout =nn.Dropout(p=0.5)

    def forward(self, enc_inputs, enc_inputs2):  #shape:[batch,node,feature]

        enc_input=enc_inputs
        if self.training:
            enc_input=torch.cat((enc_inputs,enc_inputs2),dim=0)
        
        enc_outputs, enc_self_attns = self.encoder(enc_input)
        x = enc_outputs.view(enc_outputs.size(0), -1)
        x_1 =self.dense_1(x)
        x_2 =self.dense_2(x_1)
        
        mmd_loss = 0
        if self.training:
            
            target_1 =x_1.narrow(0, enc_inputs.size(0), enc_input.size(0)-enc_inputs.size(0))
            target_2 =x_2.narrow(0, enc_inputs.size(0), enc_input.size(0)-enc_inputs.size(0))
            
            source_1=x_1.narrow(0, 0, enc_inputs.size(0))
            source_2=x_2.narrow(0, 0, enc_inputs.size(0))
          
            mmd_loss += (mmd_linear(source_1, target_1)+mmd_linear(source_2, target_2))


        return x_2,mmd_loss
      
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_linear(f_of_X, f_of_Y):  # shape: [bs, feat_channel]
    
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss
  
  