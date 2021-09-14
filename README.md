This repository contains files for "Increased Reinforcement Learning Performance through Transfer of Representation Learned by State Prediction Model" paper. 

Project Structure:
- src/: Implementation of the proposed algorithm
- gym-transfer/: A package consisting of custom made Gym environments
- scripts/: Contains scripts to generate some of the tables in the paper

Installation:
Please see "requirements.txt" for the list of required packages. Note that the code is tested only with the given versions.
After installing the packages in requirements file, install gym-transfer by `pip install -e gym-transfer`. 

Important Files:
- src/main.py: Contains 
- src/q_agent.py : Implementation of a DQN agent with option to run with Pretraining/Dualtraining methods. Most important hyperparameters are
available through arguments of main.py; please see __init__() function for further options.
- src/q_networks_list.py: Implementation of proposed network architecture.
- src/buffer_singlegoal.py : A PyTorch based replay buffer implementation.
- src/wrappers.py: State normalization functions for benchmarks used in the paper. Uses precomputed mean and std to normalize.

Important Arguments:
--alg: Chooses which DDQN algorithm will be used, options:['vanilla','pretraining','dualtraining']
--env: Choose which environment to run.
--arch_common: A list of integers determining number of units in each hidden layer in shared part.
--train_iter: Total number of training timesteps.
--pretraining_iter: Number of pretraining steps after warmup. Only used if --alg is 'pretraining'; default is 1000.
--depsilon: Exploration value.
--decay_type: Update function of exploration epsilon, options:['linear', 'exp']. If 'linear', epsilon = epsilon
--experiment_repeat: How many times run an experiment with different seeds.
--debug: Prints evaluation information etc.


An example command:

`python rl_state_prediction/main.py --debug --alg=vanilla --env='AcrobotSparse-v1' --train_iter=100000 --arch_common=[60,40,20]`

This command runs a vanilla DDQN agent on AcrobotSparse-v1 environment for 100k iterations. The network used will have 3 hidden layers
with 60, 40, and 20 units.

Note that most of the environments in gym-transfer have hyperparameters that determine reward function, observations, and some constraints in the environments. Please refer to source of code of each to see and set this hyperparameters.


Please see scripts/ for a script that can reproduce the results for CartPole.
