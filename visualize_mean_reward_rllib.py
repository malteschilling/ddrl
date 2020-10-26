import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

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

# Log file directories
exp_path = os.getenv("HOME") + '/Desktop/gpu_cluster/ray_results/test/'
experiment_dirs = [os.path.join(exp_path,dI) for dI in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path,dI))]

for i in range(0, len(experiment_dirs)):
	df = pd.read_csv(experiment_dirs[i]+'/progress.csv')
	rew_new =(df.iloc[:,2].values)
	if i==0:
		reward_values = np.vstack([rew_new])
	else:
		reward_values = np.vstack([reward_values,rew_new])

rew_mean = np.mean(reward_values, axis=0)
rew_std = np.std(reward_values, axis=0)
rew_lower_std = rew_mean - rew_std
rew_upper_std = rew_mean + rew_std

# Plotting functions
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 

#ax_arch.set_yscale('log')
#ax_arch.set_xlim(0, 500)
#ax_arch.set_ylim(0, 800)  

# Use matplotlib's fill_between() call to create error bars.   
plt.fill_between(range(0,len(rew_mean)), rew_lower_std,  
                 rew_upper_std, color=tableau20[1], alpha=0.5)  
plt.plot(range(0,len(rew_mean)), rew_mean, color=tableau20[0], lw=1)

ax_arch.set_xlabel('Episodes', fontsize=14)
ax_arch.set_ylabel('Reward per Episode', fontsize=14)

#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')

plt.show()