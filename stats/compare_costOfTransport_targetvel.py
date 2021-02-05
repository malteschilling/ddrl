import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

from scipy import stats
import scikit_posthocs as sp

"""
    Calculate statistics from evaluation runs for experiment three
    (given a specific target velocity).
    
    Four architectures are used (1 = Centralized, 2, FullyDecentral,
    3 - Local, 7 - TwoSides) for training and evaluation.
    
    Script groups results from evaluation (for each arch. 10 seeds, 100 runs), 
    calculates mean values of cost of transport 
    (for different terrains - architectures - target velocities).
    
    Input
        3_trained_cur_tvel_eval
            which is output of evaluate_trained_policies_tvel_pd.py
"""

# Set different terrain smoothness.
data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on curriculum of 
# terrains (smoothness 1.0 to 0.8),
# evaluated on 5 different uneven terrains (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 7 - two side controllers are used.
# Load data from evaluation runs.
path = os.getcwd() + '/Results/3_trained_cur_tvel_eval' # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']

# Build up data frame from all evaluations.
df_cot_eval_tv = pd.DataFrame([], columns=["approach", "seed", "cot", "smoothness", "t_vel"])
for t_vel in [1,2]:
    for j in [0,2,4]:
        for appr_i in [0,1,2,7]:
            for seed_i in range(0,10):
                mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
                    ' and seed==' + str(seed_i) + \
                    'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[appr_i] + '"')['power'])) \
                    /(8.7871 * np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
                    ' and seed==' + str(seed_i) + \
                    'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[appr_i] + '"')['distance']) )
                #mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['power'])/overall_duration) \
                 #   /(8.7871 *  mean_vel)
                new_cot_entry = pd.Series({"approach": exp_name[appr_i], 
                    "seed": seed_i, 
                    "cot": mean_cost_of_transport,
                    "smoothness": round(data_smoothn_steps[j],2),
                    "t_vel": t_vel})
                #print(new_mean_entry)
                df_cot_eval_tv = df_cot_eval_tv.append(new_cot_entry, ignore_index=True)
        

#########################################
# Statistics: Test for differences between groups
#
# We analyzed the cost of transport of the unpaired samples from the different 
# architectures using the non-parametric Kruskal-Wallis test [Kruskal & Wallis 1952] 
# (as the data appears not normally distributed) and for post-hoc analysis using the 
# Dunn [Dunn & Dunn 1961] post-hoc test (applying Bonferroni correction) 
# following [Raff 2019].
#
# We compared the difference in Cost of Transport.
#########################################
# smoothness of terrain
sm_select = 1.
# target velocity
t_vel = 1.
df_cot_eval_tv_select = df_cot_eval_tv.query('t_vel==' + str(t_vel) + ' and smoothness==' + str(sm_select))
# Run statistical test (non-parametric).
stats.kruskal( (df_cot_eval_tv_select.loc[df_cot_eval_tv_select["approach"] == exp_name[0]])["cot"],
    (df_cot_eval_tv_select.loc[df_cot_eval_tv_select["approach"] == exp_name[1]])["cot"],
    (df_cot_eval_tv_select.loc[df_cot_eval_tv_select["approach"] == exp_name[2]])["cot"],
    (df_cot_eval_tv_select.loc[df_cot_eval_tv_select["approach"] == exp_name[7]])["cot"] )
# Pair-wise comparisons between different approaches (bonferroni corrected).
sp.posthoc_dunn(df_cot_eval_tv_select, val_col='cot', group_col='approach', p_adjust = 'bonferroni')

