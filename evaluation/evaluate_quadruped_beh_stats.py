import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import glob

#import seaborn as sns
#from evaluation.compare_learning_performance_atEnd import boxplot_annotate_brackets_group

# These are the "Tableau 20" colors as RGB.    
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] 
# # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
# for i in range(len(tableau20)):    
#     r, g, b = tableau20[i]    
#     tableau20[i] = (r / 255., g / 255., b / 255.) 
#     
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams['axes.edgecolor']='#333F4B'
# plt.rcParams['axes.linewidth']=0.8
# plt.rcParams['xtick.color']='#333F4B'
# plt.rcParams['ytick.color']='#333F4B'
# 
# # Remove Type 3 fonts for latex
# plt.rcParams['pdf.fonttype'] = 42
# # matplotlib.rcParams['ps.fonttype'] = 42

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 3 - singe diag, 4 - single neig.
# 5 - two contr. diag, 6 - two neighb. contr.
# 7 - connections towards front
path = 'Results/trained_flat_eval' # use your path
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

for i in range(0,8):
    mean_cots = []
    mean_returns = []
    mean_vels = []
    for j in [0,2,4]:
        # Calculate mean velocity:
        overall_duration = np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['duration'])
        mean_return = np.mean(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['reward'])
        mean_vel = np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['distance'])/ overall_duration
        # cost_of_transport = (power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['power'])) \
            /(8.7871 * np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['distance']) )
        #mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['power'])/overall_duration) \
         #   /(8.7871 *  mean_vel)
        mean_cots.append(mean_cost_of_transport)
        mean_returns.append(mean_return)
        mean_vels.append(mean_vel)
        
    print(exp_name[i] + f' & {mean_returns[0]:.1f} & {mean_vels[0]:.3f} & {mean_cots[0]:.3f} && ' + \
        f'{mean_returns[1]:.1f} & {mean_vels[1]:.3f} & {mean_cots[1]:.3f} && ' + \
        f'{mean_returns[2]:.1f} & {mean_vels[2]:.3f} & {mean_cots[2]:.3f}')

# Centralized & 2605.9 & 0.178 & 8.224 && 1110.5 & 0.128 & 9.324 && 201.5 & 0.068 & 12.641
# FullyDecentral & 2673.0 & 0.166 & 6.283 && 974.0 & 0.107 & 7.424 && -137.5 & 0.032 & 14.099
# Local & 2943.4 & 0.177 & 6.169 && 1447.0 & 0.128 & 7.283 && 276.4 & 0.060 & 11.357
# SingleDiagonal & 2868.4 & 0.173 & 6.137 && 1185.1 & 0.114 & 7.420 && 86.7 & 0.045 & 12.341
# SingleNeighbor & 2874.8 & 0.174 & 6.244 && 1358.7 & 0.119 & 7.967 && 165.9 & 0.053 & 14.379
# SingleToFront & 2820.7 & 0.172 & 6.056 && 1229.5 & 0.116 & 7.594 && 97.2 & 0.047 & 13.033
# TwoDiags & 2786.6 & 0.175 & 7.021 && 1200.9 & 0.122 & 8.467 && 143.3 & 0.057 & 13.112
# TwoSides & 3025.6 & 0.193 & 6.758 && 1268.3 & 0.130 & 8.391 && -108.6 & 0.044 & 14.959