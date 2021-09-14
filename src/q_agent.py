
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW


import numpy as np
from buffer_singlegoal import Replay_buffer
from util import *
from q_networks_list import Q_fm
import matplotlib.pyplot as plt
criterion = nn.MSELoss()#(reduction='sum')

class Agent():
    def __init__(self,discount):
        self.discount = discount
        self.obs = None
        self.reward = None
        self.a = None
        self.done = None
        self.new_obs = None
        
    def observe(self,obs,reward,done):
        self.new_obs = obs
        self.reward = reward
        self.done = done
        
        
    def select_action(self):
        return None

class Q_agent(object):
    def __init__(self,nb_states,nb_actions,valid_actions,args,state_normalize_func):
        self.env_name = args.env
        # Hyper-parameters
        self.predict_reward = False
        self.doubleq = True
        self.optimizer = "Adam" ## Options Adam - AdamW - SGD
        self.gradient_clipping = False
        self.use_target = True
        self.tau = 0.001
        self.learning_rate = args.lr
        self.periodic_update = False and self.use_target
        self.update_rate = 5000 * self.periodic_update
        self.normalize_state_inputs_offline_metrics = True
        self.normalize_state_inputs = state_normalize_func if self.normalize_state_inputs_offline_metrics else self.identity_normalization
        self.normalize_action_flag = True
        # Pre-training
        self.warmup = args.warmup
        self.loaded_agent = (args.load_agent_path != None)
        self.pretrained_weights = False and (not self.loaded_agent)
        self.pretrained_info = "80*15_steps"
        self.pretrained_path = "work/fmresearch_bullet/work_on_buffer/Theta_Hidden_TOP_TOL/{},80,60,40, 20*2_layers-agent.pkl".format(self.pretrained_info)
        self.freeze_pretrained = False and self.pretrained_weights
        
        
        self.pretraining = (args.alg == 'pretraining') and (not self.pretrained_weights)
        
        self.train_iter_over_batch = 1       
        self.output_decay = 0.0
        self.pretraining_iter = (args.pretraining_iter + self.warmup)#*self.pretraining
        # Simultaneous Dual Output
        
        self.separate_dual = (args.alg == 'dualtraining')
        self.model_error_multiplier = 1
        self.dual_output = (False or self.separate_dual) and (not self.pretraining)
        # Normalization
        self.naive_normalize = False
        self.running_mean_normalize = False and self.dual_output
        # True: S1 - S0; False: S1
        self.state_change = True
        if self.running_mean_normalize:
            self.mean_td_error = 1
            self.mean_model_error = 1
            
        self.seed_num = args.seed
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        self.valid_actions = valid_actions
        
        n_inputs_critic = nb_states + nb_actions #+ goal_size
        n_inputs_actor = nb_states # + goal_size
        n_outputs_actor = nb_actions

        self.update_count = 0        
        self.batch_size = args.batchsize
        self.discount = args.discount
        self.decay = args.decay_func
        self.exploring_iter = args.depsilon
        self.final_eps = 0.02
        self.depsilon = (1-self.final_eps)/ self.exploring_iter
        self.depsilon_exp = np.power(self.final_eps, 1/self.exploring_iter)
        self.epsilon = 1
        self.epsilon_decay_allowed = True

        self.hiddens = args.arch_common + [-1] + args.arch_state + [-1] + args.arch_qfunc
        layers_to_freeze = [True, True] +[True, True] + [False, False]
        self.qnet = Q_fm(self.nb_states, self.nb_actions, self.hiddens, state_change = self.state_change, predict_reward = self.predict_reward)#, **net_cfg)
        if args.load_agent_path:
            self.load_weights(args.load_agent_path)
        if self.pretrained_weights:
            model_state_dict = torch.load(self.pretrained_path)
            with torch.no_grad():
                self.qnet.load_state_dict(model_state_dict)
#             self.qnet.load_weights(self.pretrained_path,self.freeze_pretrained)
# #             self.qnet.freeze_layers(layers_to_freeze)

        self.qnet_target = Q_fm(self.nb_states, self.nb_actions, self.hiddens, state_change = self.state_change, predict_reward = self.predict_reward)#, **net_cfg)
        if self.optimizer == "Adam":
            if self.output_decay != 0:
                self.qnet_optim  = Adam([{'params':self.qnet.hiddenscl.parameters()},
                                         {'params':self.qnet.hiddenso1l.parameters(),'weight_decay':self.output_decay},                                                            {'params':self.qnet.hiddenso2l.parameters()}],
                                         lr=self.learning_rate, weight_decay=0.)
            else:
                self.qnet_optim  = Adam(self.qnet.parameters(), lr=self.learning_rate)
        elif self.optimizer == "SGD":
            self.qnet_optim  = SGD(self.qnet.parameters(), lr=self.learning_rate)
        elif self.optimizer == "AdamW":
            self.qnet_optim  = AdamW(self.qnet.parameters(), lr=self.learning_rate)
    
         
        hard_update(self.qnet_target, self.qnet) # Make sure target is with the same weight
        self.buffer = Replay_buffer(args.buffersize,self.nb_states,self.nb_actions)
        
        # 
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.new_episode = False
        self.is_training = True
        # 
#         if USE_CUDA: self.cuda()
        # Keeping Track for Plotting
        self.output_path = args.output
        self.plotting = True
        self.plotallerrors = False and self.plotting
        self.plotting_freq = 5000
        self.tracking_freq = 0
        self.td_error = []
        self.model_error = []
        self.y_mean = []
        self.state_change_abs_mean = []
        
        ## Gradient Plotting
        self.plotting_gradient = False and self.plotting
#         if self.plotting_gradient:
        self.gradient_ext = None
        self.o1_gradient_abssum = []
        self.o2_gradient_abssum = []
        self.grad_l1norm = True
        self.averaging_term = 50  # How many consecutive results will be averaged
        # State
        self.atl_err1 = [] # Averaging temporary list
        self.atl_grad1 = []
        # Q-Value
        self.atl_err2 = [] # Averaging temporary list
        self.atl_grad2 = []

  
    def select_action(self, s_t, decay_epsilon=True, evaluate = False):
#         s_t[1] = 0
        s_t = self.normalize_state_inputs(s_t)
        if (not evaluate) and (np.random.uniform() < self.epsilon or (self.pretraining and (self.update_count < self.pretraining_iter))):
            action = self.random_action(self.valid_actions)
        else:
            q_list = []
            with torch.no_grad():
                for a in self.valid_actions:
                    q_list += [self.qnet(torch.tensor(np.append(s_t,self.normalize_action(a)),dtype=torch.float32))[1]]
            action = self.valid_actions[q_list.index(max(q_list))]
            
        if decay_epsilon and self.epsilon_decay_allowed:
            if self.decay == 'linear':
                self.epsilon -= self.depsilon
            else:
                self.epsilon *= self.depsilon_exp
        self.a_t = self.normalize_action(action) # return the action in the expected range but normalize it for policy update
        return action

    def observe(self,reward,obs,done):
#         super().observe(obs,reward,done)
#         self.buffer.add_entry(self.obs,self.a,self.reward,self.new_obs,int(done)) 
#         if self.is_training:    
#             if self.new_episode:
#                 self.new_episode = False
#                 self.s_t = obs
#             else:
#                 if done:
#                     self.new_episode = True
#         obs[1] = 0
        obs = self.normalize_state_inputs(obs)
        self.buffer.add_entry(self.s_t, self.a_t, reward, obs, done)
#                 self.memory.append(self.s_t, self.a_t, reward, done)

        self.s_t = obs
#         if self.is_training:    
#             self.memory.append(self.s_t, self.a_t, reward, done)
#             self.s_t = obs
            

    def random_action(self, valid_actions = None):
        if valid_actions is None:
            valid_actions = self.valid_actions
        action = np.random.choice(valid_actions)
        self.a_t = self.normalize_action(action)
        return action
    
        
    ### From DDPG IMPLEMENTATION
    def update_policy(self):
#         # Sample batch
#         state_batch, action_batch, reward_batch, \
#         next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.buffer.sample_split_batch(self.batch_size)

        state_batch = state_batch.clone().detach()#torch.tensor(state_batch,dtype=torch.float32)
        action_batch = action_batch.clone().detach()#torch.tensor(action_batch,dtype=torch.float32)
        reward_batch = reward_batch.clone().detach()#torch.tensor(reward_batch,dtype=torch.float32)
        next_state_batch = next_state_batch.clone().detach()#torch.tensor(next_state_batch,dtype=torch.float32)
        terminal_batch = terminal_batch.clone().detach()#torch.tensor(terminal_batch,dtype=torch.float32)
        state_change_batch = next_state_batch - state_batch
        if not self.state_change:
            state_change_batch = next_state_batch.clone().detach()
        if self.predict_reward:
            state_change_batch = torch.cat([state_change_batch, reward_batch],dim=1)

#         bdc = np.abs(bd-1)
        with torch.no_grad():
            q_values = None
            for a in self.valid_actions:
                action_tensor = torch.tensor([a]*next_state_batch.shape[0], dtype=torch.float32).reshape((-1,1))
                action_tensor = self.normalize_action(action_tensor)
                if q_values is None:
                    if self.use_target and not self.doubleq:
                        q_values = self.qnet_target(torch.cat([next_state_batch, action_tensor],dim=1))[1]
                    else:
                        q_values = self.qnet(torch.cat([next_state_batch, action_tensor],dim=1))[1]
                else:
                    if self.use_target and not self.doubleq:
                        q_values = torch.cat([q_values, self.qnet_target(torch.cat([next_state_batch, action_tensor],dim=1))[1]],dim=1)
                    else:
                        q_values = torch.cat([q_values, self.qnet(torch.cat([next_state_batch, action_tensor],dim=1))[1]],dim=1)
                        
            if self.doubleq:
                best_actions = q_values.argmax(dim=1).float().reshape(-1,1)
                best_actions = self.normalize_action(best_actions)
                Z = self.qnet_target(torch.cat([next_state_batch, best_actions],dim=1))[1]
            else:
                Z = q_values.max(dim=1)[0].reshape((-1,1))
            intermediary = self.discount*terminal_batch*Z
            Y = reward_batch + intermediary
            if self.naive_normalize:
                # Y = (Y - Y.mean())/(Y.std()+1e-18) #-12) ## Same normalization for inputs
#                 state_change_batch = (state_change_batch - state_change_batch.mean())/state_change_batch.std()
                state_change_batch = (state_change_batch - state_change_batch.mean(axis=0))/state_change_batch.std(axis=0)
            
        # Update Qnet
        for i in range(self.train_iter_over_batch):
            self.qnet.zero_grad()
            preds,predq = self.qnet(torch.cat([state_batch,action_batch],dim=1))#[1]
            if self.separate_dual:
                loss_qnet_td = criterion(predq,Y)
                if self.plotting and i == self.train_iter_over_batch-1:
                    self.atl_err2 = self.atl_err2 +  [loss_qnet_td.detach()]
                loss_qnet_td.backward()
                self.qnet_optim.step()
                self.qnet.zero_grad()
                preds,predq = self.qnet(torch.cat([state_batch,action_batch],dim=1))#[1]
                loss_qnet_model = criterion(preds,state_change_batch) * self.model_error_multiplier#
                if self.plotting and i == self.train_iter_over_batch-1:
                    self.atl_err1 = self.atl_err1 +  [loss_qnet_model.detach()]
                loss_qnet_model.backward()
                
                
                
            else:   
                if self.pretraining:
                    if self.update_count < self.pretraining_iter:
                        loss_qnet = criterion(preds,state_change_batch)
                        if i==self.train_iter_over_batch-1:
                            self.atl_err1 = self.atl_err1 +  [loss_qnet.detach()]
                            if self.plotallerrors:
                                self.atl_err2 = self.atl_err2 +  [criterion(predq,Y).detach()]
                            else:
                                self.atl_err2 = self.atl_err2 +  [0]
        #                 self.model_error = self.model_error + [loss_qnet.detach()]
                    else:
                        loss_qnet = criterion(predq,Y)
                        if i==self.train_iter_over_batch-1:
                            if self.plotallerrors:
                                self.atl_err1 = self.atl_err1 +  [criterion(preds,state_change_batch).detach()]
                            else:
                                self.atl_err1 = self.atl_err1 +  [0]
                            self.atl_err2 = self.atl_err2 +  [loss_qnet.detach()]            
                    loss_qnet.backward()

                else:            
                    if self.dual_output:
                        loss_qnet_td = criterion(predq,Y)
                        loss_qnet_model = criterion(preds,state_change_batch) * self.model_error_multiplier#
                        if self.running_mean_normalize:
                            loss_qnet_td = loss_qnet_td/self.mean_td_error
                            loss_qnet_model = loss_qnet_model/self.mean_model_error
                        if self.plotting and i == self.train_iter_over_batch-1:
        #                     self.td_error = self.td_error + [loss_qnet_td.detach()]
        #                     self.model_error = self.model_error + [loss_qnet_model.detach()]
                            self.atl_err1 = self.atl_err1 +  [loss_qnet_model.detach()]
                            self.atl_err2 = self.atl_err2 +  [loss_qnet_td.detach()]
                        torch.autograd.backward([loss_qnet_td,loss_qnet_model])
        #                 if self.plotting_gradient:
        #                     if self.grad_l1norm:
        #                         self.o1_gradient_abssum += [torch.sum(torch.abs(self.qnet.hiddenso1l[0].weight.grad.clone().detach()))] 
        #                         self.o2_gradient_abssum += [torch.sum(torch.abs(self.qnet.hiddenso2l[0].weight.grad.clone().detach()))] 
                    else:
                        loss_qnet = criterion(predq,Y)
                        if i == self.train_iter_over_batch-1:
                            if self.plotallerrors:
                                self.atl_err1 = self.atl_err1 +  [criterion(preds,state_change_batch).detach()]
                            else:
                                self.atl_err1 = self.atl_err1 +  [0]

                            self.atl_err2 = self.atl_err2 +  [loss_qnet.detach()]           
            #             rand = np.random.uniform()
                        loss_qnet.backward()

                if self.gradient_clipping:
                    self.qnet.gradient_clip(-1,1)
                
            self.qnet_optim.step()

            
        if self.plotting_gradient:
                                                                
            if self.grad_l1norm:
                if self.qnet.hiddenso1l[0].weight.grad is None:
                    self.atl_grad1 += [0]
                else:
                    self.atl_grad1 += [(torch.sum(torch.abs(self.qnet.hiddenso1l[0].weight.grad.clone().detach())) + 
                                      torch.sum(torch.abs(self.qnet.hiddenso1l[0].bias.grad.clone().detach()))) ] 
                    
                if self.qnet.hiddenso2l[0].weight.grad is None:
                    self.atl_grad2 += [0]
                else:
                    self.atl_grad2 += [(torch.sum(torch.abs(self.qnet.hiddenso2l[0].weight.grad.clone().detach())) + 
                                      torch.sum(torch.abs(self.qnet.hiddenso2l[0].bias.grad.clone().detach()))) ] 
            if self.update_count > 0 and self.update_count % self.averaging_term == 0:
                self.model_error += [np.mean(self.atl_err1)]
                self.td_error += [np.mean(self.atl_err2)]
                self.o1_gradient_abssum += [np.mean(self.atl_grad1)]
                self.o2_gradient_abssum += [np.mean(self.atl_grad2)]
                self.atl_err1 = []; self.atl_err2 = []; self.atl_grad1 = []; self.atl_grad2 = [];

        
        ### Normalization
        if self.running_mean_normalize:
            self.mean_td_error = self.mean_td_error + (loss_qnet_td.detach()-self.mean_td_error)/(self.update_count+1)
            self.mean_model_error = self.mean_model_error + (loss_qnet_model.detach()-self.mean_model_error)/(self.update_count+1)
            
        if self.use_target:
            if self.periodic_update:
                if self.update_count > 0 and self.update_count % self.update_rate == 0:
                    hard_update(self.qnet_target, self.qnet)
            else:
                self.soft_update()
        self.update_count =  self.update_count + 1
        
#         if self.update_count == 20000:
#             self.qnet.unfreeze_all()
        
        if self.plotting and self.update_count > 0 and self.update_count % self.plotting_freq == 0:
            self.plot()
            

    
    def reset(self, obs):
#         obs[1] = 0
        self.s_t = self.normalize_state_inputs(obs)
#         self.random_process.reset_states()

    def soft_update(self):
        self.qnet_target.soft_update(self.qnet,0.001)
        
#     def soft_update(self):      
#         #Update target networks          
#         with torch.no_grad():
#             for i in [0,2,4]:
#                 self.qnet_target[i].weight.data = (self.tau*self.qnet[i].weight.data.clone() + 
#                                                      (1-self.tau) *self.qnet_target[i].weight.data.clone())

    def plot(self):
        if self.gradient_ext is None:
            self.gradient_ext = get_exp_number(self.output_path)
#         fig = plt.figure()
#         fig.clf()
        plt.subplot(2,1,1)
        plt.plot(np.linspace(0,len(self.model_error)*self.averaging_term, len(self.model_error)),self.model_error)
        plt.xlabel(f"Iteration/{self.tracking_freq}");plt.ylabel("Error");plt.title("Model-Error")
        plt.subplot(2,1,2)
        plt.plot(np.linspace(0,len(self.td_error)*self.averaging_term, len(self.td_error)),self.td_error)
        plt.xlabel(f"Iteration/{self.tracking_freq}");plt.ylabel("Error");plt.title("TD-Error")
        plt.tight_layout()
        plot_path = self.output_path + "/Td-ModelError"
        plt.savefig(plot_path + self.gradient_ext)
#         plt.close(fig)
        if self.plotting_gradient:
#             fig = plt.figure(2)
            plt.clf()
            plt.subplot(2,1,1)
            plt.plot(np.linspace(0,len(self.o1_gradient_abssum)*self.averaging_term, len(self.o1_gradient_abssum)),
                     self.o1_gradient_abssum)
            plt.xlabel(f"Iteration/{self.tracking_freq}");plt.ylabel("magnitude");plt.title("State-Gradients-Abssum")
            plt.subplot(2,1,2)
            plt.plot(np.linspace(0,len(self.o2_gradient_abssum)*self.averaging_term, len(self.o2_gradient_abssum)),
                     self.o2_gradient_abssum)
            plt.xlabel(f"Iteration/{self.tracking_freq}");plt.ylabel("magnitude");plt.title("Q-Gradients-Abssum")
            plot_path = self.output_path + "/Gradients"
            plt.tight_layout()
            plt.savefig(plot_path + self.gradient_ext)
            to_be_saved_error = np.array([self.model_error] + [self.td_error])
            np.save(self.output_path + "/Errors_" + self.gradient_ext, to_be_saved_error)            
            to_be_saved_grad = np.array([self.o1_gradient_abssum] + [self.o2_gradient_abssum])
            np.save(self.output_path + "/Gradients_" + self.gradient_ext, to_be_saved_grad)
        plt.close('all')
    
    def save_buffer(self):
        self.buffer.save_buffer(self.output_path + "/buffer")
        
    def load_weights(self, output):
        if output is None: return

        self.qnet.load_state_dict(
            torch.load('{}/{}-agent.pkl'.format(output,self.properties()))
        )


    def properties(self):
        
        prop = (("Dual" if self.dual_output else "Single") + 
                ("sep.")*self.separate_dual +
                " O." + 
                (f"Err Mul. {self.model_error_multiplier}") * (self.dual_output and self.model_error_multiplier != 1) + 
                ((", PT: " + str(self.pretraining_iter-self.warmup) + 
                  f", WD: {self.output_decay}" * (self.output_decay != 0))*self.pretraining) +
                (", NaiveNormalize"*self.naive_normalize) +
                (", RunningMean"*self.running_mean_normalize) + ((", Upd_Rate: " + str(self.update_rate))*self.periodic_update) +
            " Arch : " + self.qnet.arch_str() +
                " " + self.optimizer + ", BS:" + str(self.batch_size) + 
                (", tau="+str(self.tau))*(self.tau != 0.001 and not self.periodic_update and self.use_target) +
               (", TI: " + str(self.train_iter_over_batch))*(self.train_iter_over_batch != 1) + #number of iterations over one batch
                 (", LR: " + str(self.learning_rate))*(self.learning_rate != 0.001) + 
               (", eps="+str(self.epsilon))*(not self.epsilon_decay_allowed) +
                (", " + 'exp_'*(self.decay != 'linear') + "deps:"+str(self.exploring_iter))*self.epsilon_decay_allowed +
                
                (", wu= " + str(self.warmup))*(self.warmup != 2000) + 
               (",DDQ")*self.doubleq +
#                 ("_pretrained" + '_' + self.pretrained_path.split('-')[1])*self.pretrained_weights +
                ("_pretrained" + '_' + self.pretrained_info)*self.pretrained_weights +
                ("_freeze")*self.freeze_pretrained +
                ("_loaded")*(self.loaded_agent) + 
                ("_offnorm")*(self.normalize_state_inputs_offline_metrics) +
                ("act")*(self.normalize_state_inputs_offline_metrics and self.normalize_action_flag) + 
                (",pred_reward")*(self.predict_reward) +
                (f"Seed:{self.seed_num}")*(self.seed_num > 0)
               ) 
            
#         prop = ("Dual: " + str(self.dual_output) + ", Running_Mean: " + str(self.running_mean_normalize) + 
#                 ", Upd_Rate: " + str(self.update_rate) )
        return prop

#     def normalize_state_inputs(self,state):
#         if self.normalize_state_inputs_offline_metrics:
# #         x, Mean:0.45, std: 1.62
# #         xdot, Mean:0.23, std: 5.67
# #         sintheta, Mean:0.06, std: 0.56
# #         costheta, Mean:0.12, std: 0.82
# #         thetadot, Mean:-0.63, std: 4.39
# #         roundtop, Mean:5.09, std: 10.51
# #         action, Mean:0.51, std: 0.50
# #             x,xdot,sintheta,costheta,thetadot,roundtop = state
#             x,x_prev,sintheta,costheta,sintheta_prev,costheta_prev,roundtop = state
# #             norm_state = np.array(((x-0.45)/1.62,(xdot-0.23)/5.67,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
# #             norm_state = np.array(((x-0.45)/1.62,(xdot-0.45)/1.62,sintheta/0.56,costheta/0.82,(thetadot+0.63)/4.4,(roundtop-5)/10.5))
#             norm_state = np.array(((x-0.45)/1.62,(x_prev-0.45)/1.62,sintheta/0.56,costheta/0.82,sintheta_prev/0.56,costheta_prev/0.82,(roundtop-5)/10.5))
#             return norm_state
#         else:
#             return state
        
    def identity_normalization(self,state):
        return state
    
    def normalize_action(self,a):
        if self.normalize_state_inputs_offline_metrics and self.normalize_action_flag:
            if "CartPole" in self.env_name or "Mountain" in self.env_name:
                return (a-0.5)*2.0
            elif "Acrobot" in self.env_name:
                return (a-1.0)/0.82
            elif "Lunar" in self.env_name:
                return (a-1.5)/1.12            
            else:
                return a
        else:
            return a

        
    def save_model(self,output):
        torch.save(
            self.qnet.state_dict(),
            '{}/{}-agent.pkl'.format(output,self.properties())
        )
#         torch.save(
#             self.critic.state_dict(),
#             '{}/critic.pkl'.format(output)
#         )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
