try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run

    assert (
        run(["pip", "install", "easypip"]).returncode == 0
    ), "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils>=0.5").setup()


import optuna
import os

import torch

from omegaconf import OmegaConf


from SACAlgo import *
        

from plot import *

from run_sac import *


import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


    

all_taux_accord = [] #recupere le taux d'accord pour chaque evaluation de la politique
steps_evaluation = [] #recupere le taux d'accord pour chaque evaluation de la politique


params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/sac-S${algorithm.seed}_${current_time:}",
    "algorithm": {
        "seed": 1,
        "n_envs": 1,
        "n_steps": 32,
        "buffer_size": 1e6,
        "batch_size": 256,
        "max_grad_norm": 0.5,
        "nb_evals": 10,
        "eval_interval": 2_000,
        "learning_starts": 10_000,
        "max_epochs": 2_000,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-7,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [22, 22],
            "critic_hidden_size": [128, 128],
        },
    },
    "gym_env": {"env_name": "CartPole-v1"},
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.00022551554188448688, 
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.0002567848671307378, 
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 0.00028917836049013965,  
    },
}

def objective(trial):
    # Sample values of alpha_critic and alpha_actor
    
    params["critic_optimizer"]["lr"] = trial.suggest_float('alpha_critic', 0.0001, 0.0003)
    params["actor_optimizer"]["lr"] = trial.suggest_float('alpha_actor', 0.0001, 0.0003)
    params["entropy_coef_optimizer"]["lr"] = trial.suggest_float('entropy_coef_optimizer',  0.0001, 0.0003)
    params["algorithm"]["architecture"]["actor_hidden_size"] = trial.suggest_categorical('actor_hidden_size', ((18, 18),(19, 19), (20, 20), (21, 21), (22, 22)))
    params["algorithm"]["architecture"]["critic_hidden_size"] = trial.suggest_categorical('critic_hidden_size', ((118, 118), (138, 138), (128, 128)))
    agents = SACAlgo(OmegaConf.create(params))
    final_reward = run_sac(agents)


    # We want to maximize the norm of the final value function

    return final_reward


"""
study_Bayes = optuna.create_study(direction='maximize')
study_Bayes.optimize(objective, n_trials=20)

# Data frame contains the params and the corresponding norm of the final value function
study_Bayes_analyse = study_Bayes.trials_dataframe(attrs=('params', 'value')) 

best_Bayes = study_Bayes.best_params
print ('The best parameters founded using Bayesian optimization are: ', best_Bayes, '\n\n')

with open("../docs/BestParam.txt", 'w', encoding='utf-8') as fichier:
    # Écrit la phrase dans le fichier
    fichier.write(str(best_Bayes))

"""
logger_nb_steps_evale = []
set_logger_reward = []
for i in range(10):
    agents = SACAlgo(OmegaConf.create(params))
    torch.manual_seed(i)
    logger_reward, logger_nb_steps_evale = run_sac(agents)
    set_logger_reward.append(logger_reward)
with open("../docs/DSACMeanReward1env.txt", 'w', encoding='utf-8') as fichier:
    # Écrit la phrase dans le fichier
    fichier.write(str(logger_nb_steps_evale) + "\n")
    for lr in set_logger_reward:
        fichier.write(str(lr) + "\n")



