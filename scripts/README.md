This folder contains a few example scripts that is provided to give some idea about how to run the experiments. Note that the most efficient way to run experiments will depend on your setup(eg. we distributed experiments on a cluster of machines).

`cartpole_experiment.sh` should take ~2 hours on a single machine. It runs all 3 algorithms on modified CartPole with Hidden Cart and Pole velocities for random seeds 1,2,3,4,5. Then, it calls `cartpole_plot.py` to create a barplot comparing 3 variants. 

    Note that it creates a directory in supplementary_material/, starting with "script-output-" and saves the runs inside. Make sure that it is empty before running an experiment.


`run_experiment.py` takes 2 arguments, env and algorithm. You can use it to run any environment-algorithm pair using the random seeds 1,2,3,4,5. 


You can adjust reward functions and observability options from the source files of the custom environments.

Note that plots in the paper was created using 80 different seeds; so, these experiments only partly reproduce results and results can vary
from the plots in the paper.