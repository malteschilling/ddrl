import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import matplotlib.patches as patches

import seaborn as sns
from evaluation.compare_learning_performance_atEnd import boxplot_annotate_brackets_group

"""
    Visualizes generalization to different target velocities
    and on changing of terrain.
    
    All trained controller had been evaluated for 10 episodes, for 
    variation of target velocity and on different uneven terrains.
    Performance is measured and plotted here
    - as a figure showing how performance develops for the architectures 
    - boxplot for specific type of terrains
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

data_smoothn_steps = np.arange(1., 0.5, -0.1) #rray([1., 0.9, 0.8, 0.7, 0.6])
data_velocity_steps = np.arange(0.5, 2.6, 0.1)
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 3 - singe diag, 4 - single neig.
# 5 - two contr. diag, 6 - two neighb. contr.
# 7 - connections towards front
path = 'Results/3_trained_cur_tvel_eval_generalization_velocity' # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']
exp_name_written = ['Centralized', 'Fully \n Decentralized', 'Local \n Neighbors', 
    'Single \n Diag. N.', 'Single \n Neigh.', 'Towards \n Front',
    'Two contr. \n diagonal', 'Two contr. \n neighbors']

# Plotting functions
####################
for smoothness_eval in data_smoothn_steps:
    fig = plt.figure(figsize=(8, 6))
    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax_arch = plt.subplot(111)  
    ax_arch.spines["top"].set_visible(False)  
    ax_arch.spines["right"].set_visible(False) 
    #ax_arch.set_yscale('log')
    ax_arch.set_xlim(0.5, 2.5)
    ax_arch.set_ylim(0., 700.)
    ax_arch.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])#, 0.5, 0.4, 0.3, 0.2,])
    ax_arch.add_patch(patches.Rectangle((0.96, 0.), 0.08, 700., color=tableau20[7]))
    ax_arch.add_patch(patches.Rectangle((1.96, 0.), 0.08, 700., color=tableau20[7]))
    #ax_arch.set_ylim(0, 800)
    for i in [0,1,2,7]:
        # Use matplotlib's fill_between() call to create error bars.   
        #plt.fill_between(data_smoothn_steps, data_min[i,:],  
        #                data_max[i,:], color=tableau20[i*2 + 1], alpha=0.25)  
        mean_val = np.zeros(len(data_velocity_steps))
        std_val = np.zeros(len(data_velocity_steps))
        for j in range(0,len(data_velocity_steps)):
            mean_val[j] = np.mean(df.query('evaluated_on==' + str(round(smoothness_eval,2)) + \
                'and target_velocity==' + str(round(data_velocity_steps[j],2)) + 'and approach=="' + exp_name[i] + '"')['reward'])
            std_val[j] = np.std(df.query('evaluated_on==' + str(round(smoothness_eval,2)) + \
                'and target_velocity==' + str(round(data_velocity_steps[j],2)) + 'and approach=="' + exp_name[i] + '"')['reward'])
        #a, b = np.polyfit(data_smoothn_steps, mean_val, deg=1)
        # Provides a regression line and the inclination of that line:
        #print(data_smoothn_steps[j], " / ", exp_name[i], " - regression: ", a)
        #ax_arch.errorbar(data_smoothn_steps, mean_val, yerr=std_val, fmt='none', ecolor=tableau20[i*2], capsize=4, capthick=2)
        ax_arch.plot(data_velocity_steps, mean_val, color=tableau20[i*2], marker='+', lw=1, label=exp_name[i])
        #print(mean_val)
    ax_arch.set_xlabel('Target velocity', fontsize=14)
    ax_arch.set_ylabel('Mean return per Episode', fontsize=14)
    file_name = "generalization_over_tvel_smoothn_" + str(round(smoothness_eval,2))
    plt.savefig(file_name + '.pdf')
    #plt.legend(loc="upper right")
    #plt.savefig(file_name + '_legend.pdf')
    #plt.show()


#########################################
# Box plot for evaluated runs
# on uneven terrain
#########################################
df_mean_eval = pd.DataFrame([], columns=["approach", "seed", "target_velocity", "evaluated_on", "mean", "std_dev"])

# Adjust target_velocity
select_appr = [0,1,2,7]
for appr_i in range(0, len(select_appr)):
    for smoothn_eval in [0.6, 0.8, 1.0]:
        for tvel_eval in [0.5, 1.0, 1.5]:
            for seed_i in range(0,10):
                select_data_p = df.query('evaluated_on==' +str(smoothn_eval) + \
                    'and seed==' + str(seed_i) + \
                    ' and target_velocity==' + str(tvel_eval) + \
                    ' and approach=="' + exp_name[select_appr[appr_i]] + '"')['reward']
                new_mean_entry = pd.Series({"approach": exp_name[select_appr[appr_i]], 
                    "seed": seed_i, 
                    "target_velocity" : tvel_eval, 
                    "evaluated_on" : smoothn_eval, 
                    "mean": np.mean(select_data_p),
                    "std_dev": np.std(select_data_p)})
                df_mean_eval = df_mean_eval.append(new_mean_entry, ignore_index=True)
          
my_pal = {}
for i in range(0,8):
    my_pal[exp_name[i]] = tableau20[i*2]

for smoothn_eval in [0.6, 0.8, 1.0]:
    for tvel_eval in [0.5, 1.0, 1.5]:
        #df_select = df.loc[df['Evaluate'].isin(['Flat','Height_010'])]
        fig_box = plt.figure(figsize=(6, 6))
        ax_mean_box = plt.subplot(111) 
        ax_mean_box.set_ylim(0., 650.)
        sns.set(style="ticks", color_codes=True)
        sns.set(context="paper", palette="colorblind", style="ticks", font_scale=1.2)
        #fig_violin = sns.FacetGrid(df_mean_eval_06, col="mean", sharey=True, size=4, aspect=.8)
        fig_box = sns.boxplot(ax=ax_mean_box, x="approach", y="mean", data=df_mean_eval.query('evaluated_on==' +str(smoothn_eval) + \
            ' and target_velocity==' + str(tvel_eval)), palette=my_pal, saturation=0.75)
        fig_box.spines['right'].set_visible(False)
        fig_box.spines['top'].set_visible(False)
        #ax_mean_box.set_xlabel('Smoothness='+str(smoothn_eval), fontsize=12)
        ax_mean_box.set_ylabel('Mean return for each seed', fontsize=14)
        fig_box.set_xticklabels(exp_name_written, fontsize=14)
        file_name = "peformance_tvel_" + str(round(tvel_eval,2)) + "_smoothn_" + str(round(smoothn_eval,2))
        plt.savefig(file_name + '.pdf')
        fig.tight_layout()

# Statistics not calculated for these data (from old data)
#xpos_box = np.arange(0,4)
#heights_box = [np.max((df_mean_eval_06_tv2.loc[df_mean_eval_06_tv2["approach"] == exp_name[i]])["mean"])-100 for i in range(0,4)]
#boxplot_annotate_brackets_group(2, [1,0], 'p < 0.01', xpos_box, heights_box)
#boxplot_annotate_brackets_group(2, [7], 'p < 0.05', xpos_box, heights_box, offset=100)
#boxplot_annotate_brackets_group(0, [1,2,3,4,5,7], 'p < 0.01', xpos_box, heights_box, offset=500)
#fig.tight_layout()
