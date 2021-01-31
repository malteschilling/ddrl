import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import seaborn as sns
from evaluation.compare_learning_performance_atEnd import boxplot_annotate_brackets_group

"""
    Visualizes variation of neural network sizes.
    
    Compared are controller architectures wrt. overall neural capacity
    - shown is mean performance 
    Horizontal axis represents size of networks.
"""

# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.) 
    
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'

# Remove Type 3 fonts for latex
plt.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

df = pd.read_csv('Results/experiment_2_nn_hidden_sizes_comparison.csv', index_col=None, header=0)

exp_name = ['Centralized', 'FullyDecentralized', 'Local', 'TwoSides']
exp_name_written = ['Centralized', 'Fully \n Decentralized', 'Local \n Neighbors', 
    'Two contr. \n neighbors']

tab_cols = [tableau20[0], tableau20[1], 
    tableau20[2], tableau20[3],
    tableau20[4], tableau20[5],
    tableau20[14], tableau20[15]]


#########################################
# Visualize trained reward for different architectures (y axis)
# compared to overall network size in neurons (x axis)
#########################################
nn_sizes = []
nn_reward_means = []
nn_reward_std = []
for appr_it in range(0,4):
    dist_nn_sizes = np.array(df.query('approach=="' + exp_name[appr_it] + '"').neurons.unique())
    dist_nn_sizes = np.sort(dist_nn_sizes)
    dist_nn_reward_mean = np.zeros( len(dist_nn_sizes) )
    dist_nn_reward_std = np.zeros( len(dist_nn_sizes) )
    for nn_it in range(0, len(dist_nn_sizes)):
        rew_nn_size = df.query('approach=="' + exp_name[appr_it] + \
            '" and neurons=="' + str(dist_nn_sizes[nn_it]) + '"')['reward']
        dist_nn_reward_mean[nn_it] = np.mean(rew_nn_size)
        dist_nn_reward_std[nn_it] = np.std(rew_nn_size)
    nn_sizes.append(dist_nn_sizes)
    nn_reward_means.append(dist_nn_reward_mean)
    nn_reward_std.append(dist_nn_reward_std)

# Plotting functions
####################
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 
ax_arch.set_xscale('log')

#ax_arch.set_yscale('log')
#ax_arch.set_xlim(1., 0.5)
#ax_arch.set_xticks([1., 0.9, 0.8, 0.7, 0.6])
#ax_arch.set_ylim(0, 800)  
for appr_it in range(3,-1,-1):
    ax_arch.errorbar(nn_sizes[appr_it], nn_reward_means[appr_it], yerr=nn_reward_std[appr_it], fmt='none', ecolor=tab_cols[appr_it*2+1], capsize=4, capthick=2)
    ax_arch.plot(nn_sizes[appr_it], nn_reward_means[appr_it], color=tab_cols[appr_it*2], marker='x', lw=2, label=exp_name_written[appr_it])

ax_arch.set_xlabel('Number of overall neurons', fontsize=12)
ax_arch.set_ylabel('Mean return per Episode', fontsize=12)
#plt.legend(loc="lower left")
#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')

#########################################
# Visualize trained reward for different architectures (y axis)
# compared to overall network size in weights (x axis)
#########################################
nn_weights = []
nn_reward_means = []
nn_reward_std = []
for appr_it in range(0,4):
    dist_nn_weights = np.array(df.query('approach=="' + exp_name[appr_it] + '"').weights.unique())
    dist_nn_weights = np.sort(dist_nn_weights)
    dist_nn_reward_mean = np.zeros( len(dist_nn_weights) )
    dist_nn_reward_std = np.zeros( len(dist_nn_weights) )
    for nn_it in range(0, len(dist_nn_weights)):
        rew_nn_size = df.query('approach=="' + exp_name[appr_it] + \
            '" and weights=="' + str(dist_nn_weights[nn_it]) + '"')['reward']
        dist_nn_reward_mean[nn_it] = np.mean(rew_nn_size)
        dist_nn_reward_std[nn_it] = np.std(rew_nn_size)
    nn_weights.append(dist_nn_weights)
    nn_reward_means.append(dist_nn_reward_mean)
    nn_reward_std.append(dist_nn_reward_std)

# Plotting functions
####################
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 
ax_arch.set_xscale('log')

#ax_arch.set_yscale('log')
#ax_arch.set_xlim(1., 0.5)
#ax_arch.set_xticks([1., 0.9, 0.8, 0.7, 0.6])
#ax_arch.set_ylim(0, 800)  
for appr_it in range(3,-1,-1):
    ax_arch.errorbar(nn_weights[appr_it], nn_reward_means[appr_it], yerr=nn_reward_std[appr_it], fmt='none', ecolor=tab_cols[appr_it*2+1], capsize=4, capthick=2)
    ax_arch.plot(nn_weights[appr_it], nn_reward_means[appr_it], color=tab_cols[appr_it*2], marker='x', lw=2, label=exp_name_written[appr_it])

ax_arch.set_xlabel('Number of overall weights', fontsize=12)
ax_arch.set_ylabel('Mean return per Episode', fontsize=12)
#plt.legend(loc="lower left")
#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')

plt.show()
