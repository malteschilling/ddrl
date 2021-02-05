import numpy as np
import pandas as pd
import glob

"""
    Summarizing evaluation runs (exp. 3) - producing mean stats, formatted for Tex.
    
    Produces:
        - Mean return
        - Mean velocities
        - Mean Cost of Transport
    for different terrains, smoothness: 1., 0.8, 0.6
    
    Input is taken from:
        3_trained_cur_tvel_eval
"""

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on uneven terrain,
# Aiming for target velocity.
# evaluated on 4 different uneven terrain (see smoothness above, 1. = flat).
# Used are only architectures 0,1,2,7
exp_name = ['Centralized', 'FullyDecentral', 'Local', 'SingleDiagonal',
       'SingleNeighbor', 'SingleToFront', 'TwoDiags', 'TwoSides']
exp_name_written = ['Centralized', 'Fully \n Decentralized', 'Local \n Neighbors', 
    'Single \n Diag. N.', 'Single \n Neigh.', 'Towards \n Front',
    'Two contr. \n diagonal', 'Two contr. \n neighbors']
path = 'Results/3_trained_cur_tvel_eval' # use your path
all_files = glob.glob(path + "/*.csv")

eval_list = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    eval_list.append(df)

df = pd.concat(eval_list, axis=0, ignore_index=True)

t_vel = 2
for i in :
    mean_cots = []
    mean_returns = []
    mean_vels = []
    for j in [0,2,4]:
        # Calculate mean velocity:
        overall_duration = np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[i] + '"')['duration'])
        mean_return = np.mean(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[i] + '"')['reward'])
        mean_vel = np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[i] + '"')['distance'])/ overall_duration
        # cost_of_transport = (power_total/steps) / (mujoco_py.functions.mj_getTotalmass(env.env.model) * com_vel)
        # Weight is 8.78710174560547
        mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[i] + '"')['power'])) \
            /(8.7871 * np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + \
            'and target_velocity=="' + str(t_vel) + '" and approach=="' + exp_name[i] + '"')['distance']) )
        #mean_cost_of_transport = (np.sum(df.query('evaluated_on==' + str(data_smoothn_steps[j]) + 'and approach=="' + exp_name[i] + '"')['power'])/overall_duration) \
         #   /(8.7871 *  mean_vel)
        mean_cots.append(mean_cost_of_transport)
        mean_returns.append(mean_return)
        mean_vels.append(mean_vel)
    print(exp_name[i] + f' & {mean_returns[0]:.1f} & {20*mean_vels[0]:.3f} & {mean_cots[0]:.3f} && ' + \
        f'{mean_returns[1]:.1f} & {20*mean_vels[1]:.3f} & {mean_cots[1]:.3f} && ' + \
        f'{mean_returns[2]:.1f} & {20*mean_vels[2]:.3f} & {mean_cots[2]:.3f}')
