# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:37:31 2021

@author: tekin.evrim.ozmermer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class SinSum(torch.nn.Module):
    def __init__(self,
                 signal_size,
                 signal_step_size,
                 number_of_signals):
        
        super(SinSum, self).__init__()
        self.signal_size = signal_size
        self.signal_step_size = signal_step_size
        self.number_of_signals = number_of_signals
        self.initialize_weights()
        self.initialize_biases()
        self.initialize_amplitude()

    def initialize_amplitude(self): # frequency
        a = torch.empty(1, self.number_of_signals)
        a_val = torch.nn.init.kaiming_normal_(a, mode='fan_out', nonlinearity='relu')
        self.a = torch.nn.Parameter(a_val, requires_grad=True) 

    def initialize_weights(self): # frequency
        w = torch.empty(1, self.number_of_signals)
        w_val = torch.nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        self.w = torch.nn.Parameter(w_val, requires_grad=True)   
    
    def initialize_biases(self): # phase shift
        b_val = torch.ones(self.number_of_signals, 1)
        b_val = torch.nn.init.kaiming_normal_(b_val, mode='fan_out', nonlinearity='relu')
        self.b = torch.nn.Parameter(b_val, requires_grad=True)

    def forward(self, test_input = None):
        if test_input==None:
            signal_input = torch.linspace(0,self.signal_size*np.pi,self.signal_step_size)
        else:
            signal_input = torch.linspace(0,self.signal_size*np.pi,self.signal_step_size)
            signal_input = torch.cat((signal_input,test_input), dim=0)
        
        self.an = self.a/torch.norm(self.a, dim=1)
        self.wn = self.w/torch.norm(self.w, dim=1)
        self.bn = self.b/torch.norm(self.b, dim=0)
        signals = self.an.T*torch.sin(self.wn.T*signal_input.unsqueeze(0)+self.bn)
        
        # signals = self.a.T*torch.sin(self.w.T*signal_input.unsqueeze(0)+self.b)
        
        return signals
