# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:28:41 2021

@author: tekin.evrim.ozmermer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



# w = torch.tensor([0,1,2,3]).unsqueeze(0)
# b = torch.tensor([0,0,0,0]).unsqueeze(0)
# signal = torch.sin(w.T*torch.arange(0, 2*np.pi, 0.1).unsqueeze(0))
# plt.plot(signal[0])
# plt.plot(signal[1])
# plt.plot(signal[2])
# plt.plot(signal[3])
# plt.plot(signal.sum(0))

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
        # b_val = torch.nn.init.kaiming_normal_(b, mode='fan_out', nonlinearity='relu')
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
        
        return signals
        
model = SinSum(signal_size = 1024, # detail size
               signal_step_size = 128, # dataset size
               number_of_signals = 2048) # number of signals to sum
                                       # bigger value -> better in the learning capability
params = [{"params": model.parameters()}]
opt = torch.optim.SGD(params, lr=0.02)
criterion = torch.nn.MSELoss()

# Training with -> signal_size = 2, signal_step_size = 20
# target = torch.linspace(0,1,model.signal_step_size)
target = torch.rand(model.signal_step_size)

steps = 1000
for cnt in range(steps):
    output = model.forward().sum(0)
    loss = criterion(output, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("STEP:", cnt, "LOSS:", loss.item())
    
    if (cnt+1)%400==0:
        plt.figure()
        plt.plot(target)
        plt.plot(output.detach())
        plt.ylim([-1, 1])
        
# Test
test_input = torch.linspace(model.signal_size,
                            model.signal_size+((model.signal_size/16)*np.pi), int(model.signal_step_size/8))
model.eval()
with torch.no_grad():
    test_output = model(test_input.detach())
model.train()
plt.figure()
plt.plot(target)
plt.plot(test_output.sum(0).detach())
plt.ylim([-1, 1])

# plt.figure()
# for elm in test_output:
#     plt.plot(elm.detach())

# Training with -> signal_size = 32, signal_step_size = 20
# model.signal_size = 32
# model.signal_step_size = 20
# target = torch.linspace(0,1,model.signal_step_size)

# steps = 10000
# for cnt in range(steps):
#     output = model.forward()
#     loss = criterion(output, target)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print("STEP:", cnt, "LOSS:", loss.item())
    
#     if (cnt+1)%2000==0:
#         plt.figure()
#         plt.plot(output.detach())
        
# # Test
# test_input = torch.linspace(0, 32*np.pi, model.signal_step_size)
# model.eval()
# with torch.no_grad():
#     test_output = model(test_input.detach())
# model.train()
# plt.figure()
# plt.plot(test_output.detach())