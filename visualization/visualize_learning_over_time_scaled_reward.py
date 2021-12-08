import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

"""
    Visualizes return during learning over time, taken from the rllib logs.
    
    Here, run on the system trained on flat terrain.
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

# Important: requires detailed logs of results (not part of the git).
exp_path = [os.getcwd() + '/results_1_global_reward/HF_10_QuantrupedMultiEnv_Centralized', #results_1_global_reward/GR_QuantrupedMultiEnv_Centralized',
    os.getcwd() + '/results_1_global_reward/HF_10_QuantrupedMultiEnv_Local',
    os.getcwd() + '/results_1_norm_reward']
   
experiment_dirs = [[os.path.join(exp_path_item,dI) for dI in os.listdir(exp_path_item) if os.path.isdir(os.path.join(exp_path_item,dI))] for exp_path_item in exp_path]

all_exp_data = []
for exp_dir in experiment_dirs:
    for i in range(0, len(exp_dir)):
        df = pd.read_csv(exp_dir[i]+'/progress.csv')
        rew_new =(df.iloc[:,2].values)
        if i==0:
            reward_values = np.vstack([rew_new])
            time_steps = (df.iloc[:,6].values)
        else:
            reward_values = np.vstack([reward_values,rew_new])
    rew_mean = np.mean(reward_values, axis=0)
    rew_std = np.std(reward_values, axis=0)
    rew_lower_std = rew_mean - rew_std
    rew_upper_std = rew_mean + rew_std
    all_exp_data.append( [rew_mean, rew_std, rew_lower_std, rew_upper_std] )
    print("Loaded ", exp_dir)

# Plotting functions
####################
# Showing results for training with the global rewards (in case of the decentralized architectures)
fig = plt.figure(figsize=(12, 8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 

#ax_arch.set_yscale('log')
ax_arch.set_xlim(0, 2e7)
#ax_arch.set_ylim(0, 800)  

# Use matplotlib's fill_between() call to create error bars.   
 
# Centralized
plt.plot(time_steps, all_exp_data[0][0], color=tableau20[0], lw=1, label=exp_path[0].split('_')[-1])
# Fully decentralized
plt.plot(time_steps, all_exp_data[1][0], color=tableau20[2], lw=1)
plt.fill_between(time_steps, 0.25*all_exp_data[2][2],  
                     0.25*all_exp_data[2][3], color=tableau20[17], alpha=0.25) 
plt.plot(time_steps, 0.25*all_exp_data[2][0], color=tableau20[16], lw=1)
ax_arch.set_xlabel('timesteps', fontsize=14)
ax_arch.set_ylabel('Return per Episode', fontsize=14)

plt.show()