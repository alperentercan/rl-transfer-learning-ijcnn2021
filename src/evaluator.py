
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None, exp_repeat = 1):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)
        
        self.exp = 0

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)

                observation, reward, done, info = env.step(action)
#                 if self.max_episode_length and episode_steps >= self.max_episode_length -1:
#                     done = True
                
                if visualize:
                   env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):
        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
#         print(f"X shape {x}, Y shape {y.shape}, Error shape {error.shape}, Results Shape {self.results.shape}")
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+ '_' + str(self.exp) + '.png')
        savemat(fn+'_' + str(self.exp) + '.mat', {'reward':self.results})
    
    def plot_all(self):
        fn = self.save_path + '/validate_reward'
        y = np.mean(self.allresults, axis=(0,1))
        error=np.std(self.allresults, axis=(0,1))
#         print(f"Results Shape : {self.results.shape}")
#         print(f"All Results Shape : {self.allresults.shape}")

        x = range(0,self.allresults.shape[2]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        plt.title(f'Average results from {self.exp} runs')
#         print(f"X shape {x}, Y shape {y.shape}, Error shape {error.shape}, All Results Shape {self.allresults.shape}")
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+ '_all' + '.png')
        savemat(fn+ '_all' + '.mat', {'reward':self.results})
        ## npy save
        np.save(fn + 'allresults',self.allresults)
        
    def new_exp(self):
        if self.exp > 0:
#             print(f"All results shape {self.allresults.shape}, results shape {self.results.shape}")
            self.allresults = np.concatenate([self.allresults,np.expand_dims(self.results,axis=0)],axis=0)
            self.results = np.array([]).reshape(self.num_episodes,0)
        else:
            self.allresults = np.expand_dims(self.results,axis=0)
            self.results = np.array([]).reshape(self.num_episodes,0)
        self.exp += 1

