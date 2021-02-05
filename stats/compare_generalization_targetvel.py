import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, os

from scipy import stats
import scikit_posthocs as sp

"""
    Calculate statistics from evaluation runs for experiment three
    (trained on curriculum of terrains (1. - 0.8 smoothness), 
    aiming for given target velocity).
    
    Four architectures are used for training and evaluation.
    
    Script groups results from evaluation (for each arch. 10 seeds, 100 runs), 
    calculates mean values for return on different terrains.
    Done for both used target velocities (1. and 2.).
    
    This produces statistics as given in Sect. 4.6
    
    Input: 
        Evaluation runs for controller (see sect. 4.6)
        3_trained_cur_tvel_eval - trained on curr. of terrains, target velocity
            output of evaluate script: evaluate_trained_policies_tvel_pd.py
"""

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures:
# evaluated on 5 different uneven terrain (see smoothness above, 1. = flat).
# In Experiment 3, only 0, 1, 2, 7 are used.
path = os.getcwd() + '/Results/3_trained_cur_tvel_eval' # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']

df_mean_eval_06_tv2 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

# Can be done for different smoothness: change evaluated on below in the queries.
# Target velocity: 2.
select_appr = [0,1,2,7]
for appr_i in range(0, len(select_appr)):
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.6 and seed==' + str(seed_i) + \
            ' and target_velocity=="2"' + \
            ' and approach=="' + exp_name[select_appr[appr_i]] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[select_appr[appr_i]], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_06_tv2 = df_mean_eval_06_tv2.append(new_mean_entry, ignore_index=True)
        
df_mean_eval_06_tv1 = pd.DataFrame([], columns=["approach", "seed", "mean", "std_dev"])

# Target velocity 1.
select_appr = [0,1,2,7]
for appr_i in range(0, len(select_appr)):
    for seed_i in range(0,10):
        select_data_p = df.query('evaluated_on==0.6 and seed==' + str(seed_i) + \
            ' and target_velocity=="1"' + \
            ' and approach=="' + exp_name[select_appr[appr_i]] + '"')['reward']
        new_mean_entry = pd.Series({"approach": exp_name[select_appr[appr_i]], 
                "seed": seed_i, 
                "mean": np.mean(select_data_p),
                "std_dev": np.std(select_data_p)})
        #print(new_mean_entry)
        df_mean_eval_06_tv1 = df_mean_eval_06_tv1.append(new_mean_entry, ignore_index=True)

#########################################
# Statistics: Test for differences between groups
#
# We analyzed the performance (mean return) of the unpaired samples from the different 
# architectures using the non-parametric Kruskal-Wallis test [Kruskal & Wallis 1952] 
# (as the data appears not normally distributed) and for post-hoc analysis using the 
# Dunn [Dunn & Dunn 1961] post-hoc test (applying Bonferroni correction) 
# following [Raff 2019].
#
# We compared the mean return for smoothness 0.6
#########################################

# For smoothness 0.6, target velocity = 2.0
####################
stats.kruskal( (df_mean_eval_06_tv2.loc[df_mean_eval_06_tv2["approach"] == exp_name[0]])["mean"],
    (df_mean_eval_06_tv2.loc[df_mean_eval_06_tv2["approach"] == exp_name[1]])["mean"],
    (df_mean_eval_06_tv2.loc[df_mean_eval_06_tv2["approach"] == exp_name[2]])["mean"],
    (df_mean_eval_06_tv2.loc[df_mean_eval_06_tv2["approach"] == exp_name[7]])["mean"] )

# KruskalResult(statistic=17.3268292682927, pvalue=0.0006053582772900576)
# k - number of groups, n - total number of observations
# eta2[H] = (H - k + 1)/(n - k), Epsilon Squared = H / ((n^2-1)/(n+1));
# eta square: 0.3980 - very strong
# epsilon square: 0.4443 - large effect
# Rea & Parker (1992) their interpretation for r, the following:
#0.36 < 0.64 - Strong
#sp.posthoc_mannwhitney(architecture_samples_at312, p_adjust = 'holm')
sp.posthoc_dunn(df_mean_eval_06_tv2, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
# >>> sp.posthoc_dunn(df_mean_eval_06_tv2, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  TwoSides
# Centralized        1.000000        1.000000  0.001441  1.000000
# FullyDecentral     1.000000        1.000000  0.002595  1.000000
# Local              0.001441        0.002595  1.000000  0.055721
# TwoSides           1.000000        1.000000  0.055721  1.000000

# For smoothness 0.6, target velocity = 1.0
####################
stats.kruskal( (df_mean_eval_06_tv1.loc[df_mean_eval_06_tv1["approach"] == exp_name[0]])["mean"],
    (df_mean_eval_06_tv1.loc[df_mean_eval_06_tv1["approach"] == exp_name[1]])["mean"],
    (df_mean_eval_06_tv1.loc[df_mean_eval_06_tv1["approach"] == exp_name[2]])["mean"],
    (df_mean_eval_06_tv1.loc[df_mean_eval_06_tv1["approach"] == exp_name[7]])["mean"] )

# KruskalResult(statistic=9.379024390243899, pvalue=0.02465378243746716)
# k - number of groups, n - total number of observations
# eta2[H] = (H - k + 1)/(n - k), Epsilon Squared = H / ((n^2-1)/(n+1));
# eta square: 0.1772 - large effect
# epsilon square: 0.2405 - relatively stron
# Rea & Parker (1992) their interpretation for r, the following:
#sp.posthoc_mannwhitney(architecture_samples_at312, p_adjust = 'holm')
sp.posthoc_dunn(df_mean_eval_06_tv1, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
# >>> sp.posthoc_dunn(df_mean_eval_06_tv1, val_col='mean', group_col='approach', p_adjust = 'bonferroni')
#                 Centralized  FullyDecentral     Local  TwoSides
# Centralized        1.000000        1.000000  0.623927  0.674307
# FullyDecentral     1.000000        1.000000  0.065768  0.073330
# Local              0.623927        0.065768  1.000000  1.000000
# TwoSides           0.674307        0.073330  1.000000  1.000000
