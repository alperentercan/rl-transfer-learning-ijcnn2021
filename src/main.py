#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym
import gym_transfer
from gym import wrappers
from time import time,localtime,strftime
import os
from evaluator import Evaluator
from q_agent import Q_agent
from util import *
import inspect
from wrappers import State_Normalizer
# gym.undo_logger_setup()

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False,save_video=False, save_agent=False, save_buffer=False):

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    done = False
    validate_flag = False
    while step < num_iterations or validate_flag:
        
        # [optional] evaluate
        if evaluate is not None and ((validate_steps > 0 and step % validate_steps == 0) or (validate_flag and done)):
            if not done:
                validate_flag = True
            else:
                validate_flag = False
                policy = lambda x: agent.select_action(x, decay_epsilon=False, evaluate=True)
                if step > num_iterations - 2*validate_steps and save_video:
                    validate_reward = evaluate(env, policy, debug=False, visualize=True)
                else:
                    validate_reward = evaluate(env, policy, debug=False, visualize=False)
                if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
                observation = None

        # reset if it is the start of episode
        if observation is None:
#             print("Resetted")
            observation = deepcopy(env.reset())
            agent.reset(observation)
            done = False

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        # env response with next_observation, reward, terminate_info
        observation2, reward, terminal, info = env.step(action)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True
        done = done or terminal
        # agent observe and update policy
        true_terminal = terminal and not 'TimeLimit.truncated' in info.keys()
        agent.observe(reward, observation2, true_terminal)
        if step > args.warmup :
            agent.update_policy()
        
#         # [optional] evaluate
#         if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
#             policy = lambda x: agent.select_action(x, decay_epsilon=False)
#             validate_reward = evaluate(env, policy, debug=False, visualize=True)
#             if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

#         # [optional] save intermideate model
#         if step % int(num_iterations/3) == 0:
#             agent.save_model(output)
        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
#         print("End of Step", done)
        if done: # end of episode
            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    if save_agent:
        agent.save_model(os.path.join(os.path.dirname(__file__),'pretrained'))
    if save_buffer:
        agent.save_buffer()

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Increased Reinforcement Learning Performance through Transfer of Representation Learned by State Prediction Model')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='CartPoleBulletPOScaled-v2', type=str, help='openai gym environment')
    parser.add_argument('--experiment_repeat', default=1, type=int, help='How many times run this experiment')
    parser.add_argument('--alg', default='vanilla',choices=['vanilla', 'pretraining', 'dualtraining'],
                        type=str, help='algorithm to be used - options: vanilla/pretraining/dualtraining', required=True)
    parser.add_argument('--train_iter', default=100000, type=int, help='Total number of training steps')
    parser.add_argument('--pretraining_iter', default=1000, type=int, help='number of pretraining steps after warmup')
    parser.add_argument('--note', default='no note', type=str, help='note for the records')
    parser.add_argument('--arch_common', default=[120,100,80,60], type=list, help='Hidden Layers of Shared Part')
    parser.add_argument('--arch_state', default=[], type=list, help='Hidden Layers of State Prediction Output Head')    
    parser.add_argument('--arch_qfunc', default=[], type=list, help='Hidden Layers of Q-Value Output Head')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=2000, type=int, help='time without training but only filling the replay buffer')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--batchsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--buffersize', default=6000000, type=int, help='Replay buffer size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during evaluation')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='maximum episode length - unless ended otherwise')
    parser.add_argument('--validate_steps', default=5000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='../output-', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--depsilon', default=25000, type=int, help='decay of exploration policy')
    parser.add_argument('--decay_func', default='linear',choices=['linear', 'exp'],
                        type=str, help='Decides epsilon decay function')    
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--save_video', default=False, type=bool, help='Save video of some simulations')
    parser.add_argument('--save_agent', default=False, type=bool, help='Save agent')
    parser.add_argument('--save_buffer', default=False, type=bool, help='Save Replay Buffer')
    parser.add_argument('--load_agent_path', default=None, type=str, help='Load agent path')

    
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') #
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()

    print(strftime("%a, %d %b %Y %H:%M:%S", localtime()))
    
    if args.env in ['AcrobotDense-v1','AcrobotSparse-v1', 'CartPoleBullet-v2', 'CartPoleBulletPO-v2'
                                                                       "CartPoleBulletPOScaled-v2","MountainCarScaled-v0"]:
        
        env = gym.make("gym_transfer:"+args.env)
    else:
        env = gym.make(args.env)

    if args.env == 'Acrobot-v2':
        env.mode_init(termination_height = 1.9)
    if args.save_video:
        env = wrappers.Monitor(env, './videos/' + args.env  + '/',force=True)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = 1 if len(env.action_space.shape) == 0 else env.action_space.shape[0]
    valid_actions = list(range(env.action_space.n))
    if args.load_agent_path:
        args.load_agent_path = os.path.join(os.path.dirname(__file__),args.load_agent_path)
        print("Agent will be loaded")
    state_normalizer = State_Normalizer(env)
    agent = Q_agent(nb_states,nb_actions,valid_actions,args,state_normalizer.normalize_state_inputs)
    
    ## Output 
    output_path = args.output + (env.parameters_str() if (args.env in ['Acrobot-v2','AcrobotDense-v1','CartPole-v2',
                                                                       'CartPoleBullet-v2',
                                                                       'CartPoleBulletPO-v2',
                                                                      'LunarLanderScaledDummy-v2',
                                                                       'LunarLanderScaled-v2',
                                                                       "CartPoleBulletPOScaled-v2"
                                                                      ]) else args.env) + "_boundaries"
    args.output = get_output_folder(output_path, args.env, agent.properties() + f", {args.train_iter} iterations")
    f = open(args.output + "/agent_init_code.txt","a")
    f.write(inspect.getsource(Q_agent.__init__))
    f.close()
    
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)
    print(args.note)
    agent.output_path = args.output
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps,args.output,# + agent.properties(),
                         max_episode_length=args.max_episode_length)
    
    for i in range(args.experiment_repeat):

        
        if args.mode == 'train':
            train(args.train_iter, agent, env, evaluate, 
                args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug,save_video=args.save_video,save_agent=args.save_agent, save_buffer=args.save_buffer)

        elif args.mode == 'test':
            test(args.validate_episodes, agent, env, evaluate, args.resume,
                visualize=True, debug=args.debug)

        else:
            raise RuntimeError('undefined mode {}'.format(args.mode))
        
        evaluate.new_exp()
        agent = Q_agent(nb_states,nb_actions,valid_actions,args,state_normalizer.normalize_state_inputs)
    if args.experiment_repeat > 1:
        evaluate.plot_all()
    print("Finished")
    env.close()