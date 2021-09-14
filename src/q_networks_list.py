
import torch
import numpy as np
import os

'''
Implementation of proposed network architecture in "Increased Reinforcement Learning Performance through Transfer of Representation Learned by State Prediction Model" paper.

'''

# A weight initialization scheme.
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Q_fm(torch.nn.Module):
    
    def __init__(self,nb_states, nb_actions, n_hiddens, init_w=3e-3, state_change=True, predict_reward=False):
        '''
        nb_states(int): Number of dimensions for MDP states
        nb_actions(int): Number of dimensions for MDP actions
        n_hiddens(list): A list of integers indicating number of units per hidden layer. Starts with layers of common part; then, layers
        of state output head; finally, layers of Q-value head Expected to contain two -1 entries, one seperating layers of common part
        from state head's, another seperating layers of state head from Q-value head's. If a part doesn't have any hidden layers, should
        be skipped.
        init_w(float): Magnitude for random initialization of output layers.
        state_change(bool): Whether state output head predicts state change or next state. For backward compatiblitiy and logging only - 
        appropriate data should be given when when forward function is called.
        predict_reward(bool): Whether state output head should predict immediate reward too. Used to adjust size of output layer, output 
        batch should be chosen according to.
        '''
        super(Q_fm, self).__init__()
        i1 = n_hiddens.index(-1)
        i2 = n_hiddens.index(-1,i1 + 1)
        
        hiddensc = n_hiddens[:i1]
        if predict_reward:
            hiddenso1 = n_hiddens[i1+1:i2] + [nb_states + 1]
        else:
            hiddenso1 = n_hiddens[i1+1:i2] + [nb_states]
        hiddenso2 = n_hiddens[i2 + 1:] + [1]

        # Common
        self.hiddenscl =  [torch.nn.Linear(nb_states + nb_actions, hiddensc[0])]
        for i in range(len(hiddensc)-1):
            self.hiddenscl += [torch.nn.Linear(hiddensc[i], hiddensc[i+1])]
        self.hiddenscl = torch.nn.ModuleList(self.hiddenscl)
        # State Output
        self.hiddenso1l = [torch.nn.Linear(hiddensc[-1], hiddenso1[0])]
        for i in range(len(hiddenso1)-1):
            self.hiddenso1l += [torch.nn.Linear(hiddenso1[i], hiddenso1[i+1])]
        self.hiddenso1l = torch.nn.ModuleList(self.hiddenso1l)

        # Q Output
        self.hiddenso2l = [torch.nn.Linear(hiddensc[-1], hiddenso2[0])]
        for i in range(len(hiddenso2)-1):
            self.hiddenso2l += [torch.nn.Linear(hiddenso2[i], hiddenso2[i+1])]
        self.hiddenso2l = torch.nn.ModuleList(self.hiddenso2l)
        self.hiddensc = hiddensc
        self.hiddenso1 = hiddenso1
        self.hiddenso2 = hiddenso2
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.init_weights(init_w)
        self.state_change = state_change
        
        
    def init_weights_all(self):
        for layer in self.hiddenscl:
            layer.weight.data = fanin_init(layer.weight.data.size())
        for i in range(len(self.hiddenso1l)):
            layer = self.hiddenso1l[i]
            layer.weight.data = fanin_init(layer.weight.data.size())
        for j in range(len(self.hiddenso2l)):
            layer = self.hiddenso2l[j]
            layer.weight.data = fanin_init(layer.weight.data.size())    
            
    def init_weights(self, init_w):
        for layer in self.hiddenscl:
            layer.weight.data = fanin_init(layer.weight.data.size())
        for i in range(len(self.hiddenso1l)-1):
            layer = self.hiddenso1l[i]
            layer.weight.data = fanin_init(layer.weight.data.size())
        for j in range(len(self.hiddenso2l)-1):
            layer = self.hiddenso2l[j]
            layer.weight.data = fanin_init(layer.weight.data.size())    
                       
        self.hiddenso1l[-1].weight.data.uniform_(-init_w, init_w)
        self.hiddenso2l[-1].weight.data.uniform_(-init_w, init_w)

    def load_weights(self,path,freeze_loaded=False):
        '''
        Function to load weights from a PyTorch file.
        path(string): Relative path from this file.
        freeze_loaded(bool): Freezes loaded weights only - ie. no gradient will be calculated for those. 
        '''
        direc = os.path.dirname(os.path.abspath(__file__))
        weightsTBL = torch.load(os.path.join(direc,path))
#         assert len(weightsTBL.keys()) > 2*(len(self.hiddensc) + len(self.hiddenso1))
        i = 0
        modellist = list(weightsTBL.values())
        for param in self.hiddenscl:
            if i <= len(modellist) - 2:
                if param.weight.size() == modellist[i].size():
                    print(f"Copied common layer with size: {modellist[i].size()}")
                    param.weight.data.copy_(modellist[i])
                    if freeze_loaded:
                        param.weight.requires_grad = False
                if param.bias.size() == modellist[i+1].size():
                    param.bias.data.copy_(modellist[i+1])
                    if freeze_loaded:
                        param.bias.requires_grad = False
            i += 2
        for param in self.hiddenso1l:
            if i <= len(modellist) - 2:
                if param.weight.size() == modellist[i].size():
                    print(f"Copied output layer with size: {modellist[i].size()}")
                    param.weight.data.copy_(modellist[i])
                    if freeze_loaded:
                        param.weight.requires_grad = False
                if param.bias.size() == modellist[i+1].size():
                    param.bias.data.copy_(modellist[i+1])
                    if freeze_loaded:
                        param.bias.requires_grad = False
            i += 2
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def freeze_layers(self,layers):#[0,0,1] +  [1] +  [1] 
        '''
        Expects a list of booleans of length number of layers. Each boolean controls gradient computation of corresponding layer.
        '''
        assert len(layers) == len(self.hiddensc) + len(self.hiddenso1) + len(self.hiddenso2)
        indexes = [(0,len(self.hiddensc)),
                   (len(self.hiddensc),len(self.hiddensc) + len(self.hiddenso1))
                   ,(len(self.hiddensc) + len(self.hiddenso1),len(layers))]
        parts = [self.hiddenscl,self.hiddenso1l,self.hiddenso2l]
        for i in range(len(parts)):
            for freeze,param in zip(layers[indexes[i][0]:indexes[i][1]],parts[i]):
                if freeze:
                    param.weight.requires_grad = False
                    param.bias.requires_grad = False
            

    def unfreeze_layers(self,layers):#[0,0,1] +  [1] +  [1] 
        assert len(layers) == len(self.hiddensc) + len(self.hiddenso1) + len(self.hiddenso2)
        for unfreeze,param in zip(layers,self.hiddenscl + self.hiddenso1l + self.hiddenso2l):
            if unfreeze:
                param.weight.requires_grad = True
                param.bias.requires_grad = True
            
    def forward(self, x):
#         x = torch.cat([state,action],dim=1)
        # Common
        for layer in self.hiddenscl:
            x = layer(x)
            x = self.relu(x)
        # State Output
        s = self.hiddenso1l[0](x)        
        for layer in self.hiddenso1l[1:]:
            s = self.relu(s)
            s = layer(s)
        # Q Output
        q = self.hiddenso2l[0](x)        
        for layer in self.hiddenso2l[1:]:
            q = self.relu(q)
            q = layer(q)   
        return s, q
    
    def soft_update(self, source, tau):
        for param, source_param in zip(self.parameters(), source.parameters()):
            param.data.copy_(
                param.data * (1.0 - tau) + source_param.data * tau
            )   
    def gradient_clip(self,ming, maxg):
        for param in self.parameters():
            param.grad.data.clamp_(ming, maxg)
            
    def arch_str(self):
        return f"Common: {self.hiddensc}, " + ("State Change: " if self.state_change else "Next State: ") + f"{self.hiddenso1}, Q: {self.hiddenso2}"
