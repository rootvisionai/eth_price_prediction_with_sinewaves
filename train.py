# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:38:53 2021

@author: tekin.evrim.ozmermer
"""

import torch
from models import SinSum
import utils
import matplotlib.pyplot as plt

# import data
df = utils.import_data("datasets/ETHUSDT_Binance_futures_data_hour.csv")
df = df.reindex(index=df.index[::-1])
train_set, full_set = utils.prepare_data(df, column = "close",
                                         training_data_ratio = 1)

# prepare model
model = SinSum(signal_size = train_set.shape[0], # detail size 128
               signal_step_size = train_set.shape[0], # dataset size
               number_of_signals = 1024*10) # number of signals to sum 256
                                       # bigger value -> better in the learning capability
params = [{"params": model.parameters()}]

# prepare optimizer and loss function
opt = torch.optim.Adam(params, lr=0.01)
criterion = torch.nn.MSELoss()

target = train_set
plt.figure()
plt.plot(target, label='TARGET')

steps = 100
for cnt in range(steps):
    output = model.forward().sum(0)
    loss = criterion(output.float(), target.float().squeeze(1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("STEP:", cnt, "LOSS:", loss.item())
    
    if (cnt+1)%10==0:
        plt.plot(output.detach(), label='STEP-{}'.format(cnt+1))
        plt.legend()
        
test_input = torch.linspace(train_set.shape[0],
                            full_set.shape[0], # model.signal_size+((model.signal_size/2)*np.pi),
                            int(full_set.shape[0]-train_set.shape[0])) # int(model.signal_step_size/2))
model.eval()
with torch.no_grad():
    test_output = model(test_input.detach())
model.train()
vis_set = torch.tensor([(elm+full_set[cnt-1,0]+full_set[cnt-2,0]+full_set[cnt-3,0])/4 for cnt, elm in enumerate(full_set[:,0]) if cnt > 3])

plt.figure()
plt.axvline(train_set.shape[0],c="g")
plt.plot(vis_set)
plt.plot(test_output.sum(0).detach())
plt.ylim([-1, 1])

from simulator import Buyer

agent = Buyer(test_output.sum(0)[train_set.shape[0]:],
              full_set[train_set.shape[0]:],
              usd = 100, eth = 100, r = 10, amount=1)

capital = agent.simulate()
plt.figure()
plt.plot([elm for elm in capital.keys()],[elm for elm in capital.values()])
plt.scatter([elm for elm in capital.keys()],[elm for elm in capital.values()])

### PREDICTION PART ###

test_input = torch.linspace(train_set.shape[0],
                            train_set.shape[0] + 1024, # model.signal_size+((model.signal_size/2)*np.pi),
                            1024) # int(model.signal_step_size/2))
model.eval()
with torch.no_grad():
    test_output = model(test_input.detach())
model.train()
vis_set = torch.tensor([(elm+full_set[cnt-1,0]+full_set[cnt-2,0]+full_set[cnt-3,0])/4 for cnt, elm in enumerate(full_set[:,0]) if cnt > 3])

plt.figure()
plt.axvline(train_set.shape[0],c="g")
plt.plot(vis_set)
plt.plot(test_output.sum(0).detach())
plt.ylim([-1, 1])

from simulator import Buyer

agent = Buyer(test_output.sum(0)[train_set.shape[0]:],
              test_output.sum(0)[train_set.shape[0]:],
              usd = 100, eth = 100, r = 10, amount=1)

capital = agent.simulate()
plt.figure()
plt.plot([elm for elm in capital.keys()],[elm for elm in capital.values()])
plt.scatter([elm for elm in capital.keys()],[elm for elm in capital.values()])