import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

from scipy import stats
import scikit_posthocs as sp

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 3 - singe diag, 4 - single neig.
# 5 - two contr. diag, 6 - two neighb. contr.
# 7 - connections towards front
path = 'Results/3_trained_cur_tvel_eval' # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']


# Target velocity: 1.
df_cot_eval_06_tv = pd.DataFrame([], columns=["approach", "seed", "cot", "smoothness", "t_vel"])
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
                df_cot_eval_06_tv = df_cot_eval_06_tv.append(new_cot_entry, ignore_index=True)
        

#########################################
# Statistics: Test for differences between groups
#########################################
sm_select = 1.
t_vel = 1
df_cot_eval_06_tv_select = df_cot_eval_06_tv.query('t_vel==' + str(t_vel) + ' and smoothness==' + str(sm_select))
stats.kruskal( (df_cot_eval_06_tv_select.loc[df_cot_eval_06_tv_select["approach"] == exp_name[0]])["cot"],
    (df_cot_eval_06_tv_select.loc[df_cot_eval_06_tv_select["approach"] == exp_name[1]])["cot"],
    (df_cot_eval_06_tv_select.loc[df_cot_eval_06_tv_select["approach"] == exp_name[2]])["cot"],
    (df_cot_eval_06_tv_select.loc[df_cot_eval_06_tv_select["approach"] == exp_name[7]])["cot"] )
sp.posthoc_dunn(df_cot_eval_06_tv_select, val_col='cot', group_col='approach', p_adjust = 'bonferroni')

