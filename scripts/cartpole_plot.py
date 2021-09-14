import matplotlib.pyplot as plt
import os
import numpy as np


def plot_cartpole():
    from scipy.io import loadmat
    output_folder = "../script-output-CartPolePOScaled : Hide Xdot, Hide thetadot, _boundaries"
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    
    pt= []
    nopt = []
    dual = []
    
    for seed in range(1,6):
        for file in os.listdir(output_folder):
            if f"Seed:{seed}," in file:
                path = os.path.join(output_folder,file)
                if "PT" in file:
                    load = np.array(loadmat(os.path.join(path,"validate_reward_0.mat"))['reward'])
                    pt.append(np.mean(load, axis=0))
                elif "Dual" in file:
                    load = np.array(loadmat(os.path.join(path,"validate_reward_0.mat"))['reward'])
                    dual.append(np.mean(load, axis=0))            
                elif "Single" in file:
                    load = np.array(loadmat(os.path.join(path,"validate_reward_0.mat"))['reward'])
                    nopt.append(np.mean(load, axis=0))                            
        

    func = np.median
    ax.bar(0,func(np.array(nopt), axis=0)[14], label="NoPT", color='k')         
    ax.bar(1,func(np.array(pt), axis=0)[14], label="PT", color='r')   
    ax.bar(2,func(np.array(dual), axis=0)[14], label="Dual", color='b')         
    ax.legend()
    plt.savefig("../CartPoleFigure.png")
        
if __name__ == "__main__":
    plot_cartpole()
