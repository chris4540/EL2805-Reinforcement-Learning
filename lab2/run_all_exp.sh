#!/bin/bash

# check memory size on learning performance
python cartpole_dqn.py --men_size 5000 --folder mem_size_5000
python cartpole_dqn.py --men_size 10000 --folder mem_size_10000
python cartpole_dqn.py --men_size 20000 --folder mem_size_20000

# check learning rate
python cartpole_dqn.py --lr 0.005 --folder lr_5E-3
python cartpole_dqn.py --lr 0.001 --folder lr_1E-3
python cartpole_dqn.py --lr 0.01  --folder lr_1E-2

# check update freq
python cartpole_dqn.py --update_fq 1 --folder update_fq_1
python cartpole_dqn.py --update_fq 5 --folder update_fq_5
python cartpole_dqn.py --update_fq 10 --folder update_fq_10