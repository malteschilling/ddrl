import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

from visualization.catscatter import catscatter

# Plotting Presets
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

#####################################
# Loading (normalized) gradient data
#####################################
# i.e. this is the direct calculated change around the original values from the 
# policy network - variation is proportional to std. dev. which normalizes this 
# wrt. to the specific sensory channel
# Along the DRL rollout trajectory all this gradient values are summed 
# (we are using the absolute value of gradient as an indicator).

#manual_grads = np.load("grads_3.npy")
manual_grads_abs = np.load("computed_grads/grads_tvel1_abs_8.npy")
norm_grads_abs = manual_grads_abs / np.sum(manual_grads_abs,axis=0)



#####################################
# Produce different kinds of connection plots
#####################################
# Simple weight matrix plot as imshow
fig, ax = plt.subplots(figsize=(4, 8))
(ax.imshow(norm_grads_abs, interpolation='none',norm=LogNorm(), cmap=cm.Greys)).set_rasterized(True)
ax.set_aspect('auto')
ax.set_xticks(np.arange(8))

# Use categorical scatter plot
df = pd.DataFrame([], columns=["row", "col", "strength"])
log_grads_abs = (np.log(norm_grads_abs) - np.min(np.log(norm_grads_abs)))
for i in range(0,log_grads_abs.shape[0]):
    for j in range(0,log_grads_abs.shape[1]):
        new_pd_entry = pd.Series({"row": i, 
                "column": j, 
                "strength": log_grads_abs[i,j]})
        print(i,j)
        df = df.append(new_pd_entry, ignore_index=True)


#####################################
# Differentiate different contributions:
# [global information, FL first joint, FL second joint, HL 2 joints, HR 2 joints, FR 2 joints]
contribution_idx = [[0,1,2,3,4,13,14,15,16,17,18,43],\
             [11,25,33,35],\
             [12,26,34,36],\
             [ 5,19,27,37],\
             [ 6,20,28,38],\
             [ 7,21,29,39],\
             [ 8,22,30,40],\
             [ 9,23,31,41],\
             [10,24,32,42]]
# Group gradient information in contributions.
contributions = np.zeros((9,8))
for act_i in range(0,8):
    for contr_i in range(0,9):
        sum_contr = 0.
        for el_contr in contribution_idx[contr_i]:
            sum_contr += norm_grads_abs[el_contr, act_i]
        contributions[contr_i,act_i] = sum_contr        

#####################################
# Shows for each individual joint the contribution
#####################################
ind = np.arange(8)
width = 0.5
plt.bar(ind, contributions[-1,:], width)
for i in range(0,8):
    plt.bar(ind, contributions[-(i+2),:], width, bottom=np.sum(contributions[-(i+1):,:],axis=0))
plt.xticks(ind, ('FR', '', 'FL', '', 'HL', '', 'HR'))

#####################################
# Group according to locality (global, same joint, same leg, neighb. leg, diag. leg)
#####################################
fig = plt.figure(figsize=(4, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_one = plt.subplot(1,2,1)  
ax_one.spines["top"].set_visible(False)  
ax_one.spines["right"].set_visible(False) 
ax_one.spines["left"].set_visible(False) 
ax_one.spines["bottom"].set_visible(False) 
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#ax_arch.set_xscale('log')
#ax_arch.set_xlabel('', fontsize=12)

# Distribute the contributions of the joints into the groups:
# (global, same joint, same leg, neighb. leg, diag. leg)
contrib_joints = np.zeros((2,5))
contrib_joints[:,0] = np.mean(contributions[0])
contrib_joints[0,1] = np.mean(contributions[[1,3,5,7],[0,2,4,6]])
contrib_joints[1,1] = np.mean(contributions[[2,4,6,8],[1,3,5,7]])
contrib_joints[1,2] = np.mean(contributions[[1,3,5,7],[1,3,5,7]])
contrib_joints[0,2] = np.mean(contributions[[2,4,6,8],[0,2,4,6]])
contrib_joints[0,3] = np.mean([np.sum(contributions[[3,4,7,8],[0,0,0,0]]),\
    np.sum(contributions[[1,2,5,6],[2,2,2,2]]),\
    np.sum(contributions[[3,4,7,8],[4,4,4,4]]),\
    np.sum(contributions[[1,2,5,6],[6,6,6,6]])])
contrib_joints[1,3] = np.mean([np.sum(contributions[[3,4,7,8],[1,1,1,1]]),\
    np.sum(contributions[[1,2,5,6],[3,3,3,3]]),\
    np.sum(contributions[[3,4,7,8],[5,5,5,5]]),\
    np.sum(contributions[[1,2,5,6],[7,7,7,7]])])
contrib_joints[0,4] = np.mean([np.sum(contributions[[5,6],[0,0]]),\
    np.sum(contributions[[7,8],[2,2]]),\
    np.sum(contributions[[1,2],[4,4]]),\
    np.sum(contributions[[3,4],[6,6]])])
contrib_joints[1,4] = np.mean([np.sum(contributions[[5,6],[1,1]]),\
    np.sum(contributions[[7,8],[3,3]]),\
    np.sum(contributions[[1,2],[5,5]]),\
    np.sum(contributions[[3,4],[7,7]])])
contrib_joints_norm = contrib_joints.T/np.sum(contrib_joints,axis=1)
contrib_joints = np.sum(contrib_joints,axis=0)/2.

# Summarize information from different legs.
ind = [0., 1.]
width=0.5
# When differentiating the two joints of a leg this code was used.
# col_list = [tableau20[6],tableau20[4],tableau20[3],tableau20[2],tableau20[0]]
# plt.bar(ind, contrib_joints_norm[-1,:], width, color=col_list[0],edgecolor='black')
# for i in range(0,4):
#     plt.bar(ind, contrib_joints_norm[-(i+2),:], width, bottom=np.sum(contrib_joints_norm[-(i+1):,:],axis=0), color=col_list[i+1],edgecolor='black')

ind = [0.]
# Showing a baseline for equal distribution.
ax_one.set_title('Equal distribution \n of features')
contrib_equal = np.array([12,4,4,16,8])
contrib_equal = contrib_equal/np.sum(contrib_equal)
plt.bar(ind, contrib_equal[4], width, color=col_list[0],edgecolor='black')
plt.bar(ind, contrib_equal[3], width, bottom=np.sum(contrib_equal[-1:],axis=0), color=col_list[1],edgecolor='black')
plt.bar(ind, 0.5*contrib_equal[3], width, bottom=np.sum(contrib_equal[-1:],axis=0), color=col_list[1],edgecolor='black')
plt.bar(ind, contrib_equal[2], width, bottom=np.sum(contrib_equal[-2:],axis=0), color=col_list[2],edgecolor='black')
plt.bar(ind, contrib_equal[1], width, bottom=np.sum(contrib_equal[-3:],axis=0), color=col_list[3],edgecolor='black')
plt.bar(ind, contrib_equal[0], width, bottom=np.sum(contrib_equal[-4:],axis=0), color=col_list[4], edgecolor='black')

ax_two = plt.subplot(1,2,2)  
ax_two.set_title('Learned distribution \n of features')
ax_two.spines["top"].set_visible(False)  
ax_two.spines["right"].set_visible(False) 
ax_two.spines["left"].set_visible(False) 
ax_two.spines["bottom"].set_visible(False) 
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.bar(ind, contrib_joints[4], width, color=col_list[0],edgecolor='black')
plt.bar(ind, contrib_joints[3], width, bottom=np.sum(contrib_joints[-1:],axis=0), color=col_list[1],edgecolor='black')
# For better visual comparison: show two halfs for the two neighboring legs.
plt.bar(ind, 0.5*contrib_joints[3], width, bottom=np.sum(contrib_joints[-1:],axis=0), color=col_list[1],edgecolor='black')
plt.bar(ind, contrib_joints[2], width, bottom=np.sum(contrib_joints[-2:],axis=0), color=col_list[2],edgecolor='black')
plt.bar(ind, contrib_joints[1], width, bottom=np.sum(contrib_joints[-3:],axis=0), color=col_list[3],edgecolor='black')
plt.bar(ind, contrib_joints[0], width, bottom=np.sum(contrib_joints[-4:],axis=0), color=col_list[4], edgecolor='black')

#####################################
# Differentiate between kind of information
#####################################
# Sum up for hip and knee joins:
hip_grad_abs = np.sum(norm_grads_abs[:,[0,2,4,6]], axis=1)
knee_grad_abs = np.sum(norm_grads_abs[:,[1,3,5,7]], axis=1)

contribution_type_idx = [[0,1,2,3,4,14,15,16,17,18],\
    [13],\
    [43],\
    range(5,13),\
    range(19,27),\
    range(27,35),\
    range(35,43)]
             
# Group gradient information in contributions.
# Rest global, x_vel, t_vel, joint angles, joint vel., passive torques, last action
contributions_type = np.zeros((7,2))
for contr_i in range(0,7):
    sum_contr_knee = 0.
    sum_contr_hip = 0.
    for el_contr in contribution_type_idx[contr_i]:
        sum_contr_hip += hip_grad_abs[el_contr]
        sum_contr_knee += knee_grad_abs[el_contr]
    contributions_type[contr_i,0] = sum_contr_hip
    contributions_type[contr_i,1] = sum_contr_knee
contributions_type = contributions_type/np.sum(contributions_type)

ind = [0., 1.]
width=0.5
# When differentiating the two joints of a leg this code was used.
plt.bar(ind, contributions_type[-1,:], width, color=tableau20[0],edgecolor='black')
for i in range(1,7):
    plt.bar(ind, contributions_type[-(i+1),:], width, bottom=np.sum(contributions_type[-i:,:],axis=0), color=tableau20[2*i],edgecolor='black')

plt.show()

# TVEL 1., Smoothness 1.
# Mean for  Centralized / 0 : 317.95, std.dev.: 14.21, Distance: 54.51, Steps: 1000.00
# Mean for  Centralized / 1 : 419.45, std.dev.: 13.07, Distance: 52.24, Steps: 1000.00
# Mean for  Centralized / 3 : 487.21, std.dev.: 9.05, Distance: 50.57, Steps: 1000.00
# Mean for  Centralized / 2 : 566.77, std.dev.: 8.63, Distance: 49.14, Steps: 1000.00
# Mean for  Centralized / 4 : 438.15, std.dev.: 9.72, Distance: 52.90, Steps: 1000.00
# Mean for  Centralized / 0 : 392.38, std.dev.: 8.71, Distance: 48.15, Steps: 1000.00
# Mean for  Centralized / 1 : 545.22, std.dev.: 6.85, Distance: 50.60, Steps: 1000.00
# Mean for  Centralized / 3 : 461.70, std.dev.: 5.52, Distance: 52.85, Steps: 1000.00
# Mean for  Centralized / 2 : 547.87, std.dev.: 12.74, Distance: 50.70, Steps: 1000.00
# Mean for  Centralized / 4 : 422.38, std.dev.: 7.03, Distance: 52.94, Steps: 1000.00
# Overall Mean for  Centralized : 459.91, std.dev.: 75.10; CoT: 9.74; Vel.: 0.05
# 
# Mean for  Centralized / 0 : 262.22, std.dev.: 21.26, Distance: 56.61, Steps: 1000.00
# Mean for  Centralized / 1 : 377.82, std.dev.: 15.46, Distance: 54.06, Steps: 1000.00
# Mean for  Centralized / 3 : 455.90, std.dev.: 13.40, Distance: 50.32, Steps: 1000.00
# Mean for  Centralized / 2 : 502.32, std.dev.: 36.90, Distance: 45.65, Steps: 1000.00
# Mean for  Centralized / 4 : 386.64, std.dev.: 21.77, Distance: 53.95, Steps: 1000.00
# Mean for  Centralized / 0 : 367.43, std.dev.: 17.35, Distance: 48.16, Steps: 1000.00
# Mean for  Centralized / 1 : 535.60, std.dev.: 7.99, Distance: 49.65, Steps: 1000.00
# Mean for  Centralized / 3 : 442.36, std.dev.: 14.53, Distance: 52.89, Steps: 1000.00
# Mean for  Centralized / 2 : 521.17, std.dev.: 6.05, Distance: 51.70, Steps: 1000.00
# Mean for  Centralized / 4 : 386.16, std.dev.: 14.32, Distance: 53.81, Steps: 1000.00
# Overall Mean for  Centralized : 423.76, std.dev.: 82.00; CoT: 9.86; Vel.: 0.05
# TVel  && 423.76 & (82.00) && 0.05168073632635414 & (0.00) & 9.86
# 
# TVEL 1., Smoothness 0.8
# Mean for  Centralized / 0 : 283.88, std.dev.: 40.99, Distance: 51.71, Steps: 952.40
# Mean for  Centralized / 1 : 281.96, std.dev.: 23.70, Distance: 56.87, Steps: 1000.00
# Mean for  Centralized / 3 : 457.46, std.dev.: 11.07, Distance: 51.60, Steps: 1000.00
# Mean for  Centralized / 2 : 521.07, std.dev.: 14.72, Distance: 46.62, Steps: 1000.00
# Mean for  Centralized / 4 : 392.01, std.dev.: 14.50, Distance: 53.85, Steps: 1000.00
# Mean for  Centralized / 0 : 357.93, std.dev.: 23.90, Distance: 48.21, Steps: 1000.00
# Mean for  Centralized / 1 : 528.65, std.dev.: 5.55, Distance: 51.54, Steps: 1000.00
# Mean for  Centralized / 3 : 430.50, std.dev.: 76.89, Distance: 49.54, Steps: 946.80
# Mean for  Centralized / 2 : 514.81, std.dev.: 9.36, Distance: 51.87, Steps: 1000.00
# Mean for  Centralized / 4 : 407.90, std.dev.: 18.27, Distance: 53.66, Steps: 1000.00
# Overall Mean for  Centralized : 417.62, std.dev.: 91.95; CoT: 9.88; Vel.: 0.05
# TVel  && 417.62 & (91.95) && 0.052071092344330765 & (0.00) & 9.88
# 
# TVEL 2., Smoothness 1.
# Mean for  Centralized / 0 : 129.15, std.dev.: 35.44, Distance: 78.25, Steps: 969.00
# Mean for  Centralized / 1 : 336.33, std.dev.: 128.05, Distance: 94.74, Steps: 1000.00
# Mean for  Centralized / 3 : 210.20, std.dev.: 22.68, Distance: 90.40, Steps: 1000.00
# Mean for  Centralized / 2 : 460.84, std.dev.: 9.44, Distance: 95.29, Steps: 1000.00
# Mean for  Centralized / 4 : 352.84, std.dev.: 43.96, Distance: 97.97, Steps: 1000.00
# Mean for  Centralized / 0 : 58.05, std.dev.: 27.93, Distance: 54.33, Steps: 812.50
# Mean for  Centralized / 1 : 286.67, std.dev.: 12.08, Distance: 88.62, Steps: 1000.00
# Mean for  Centralized / 3 : 10.16, std.dev.: 46.04, Distance: 26.97, Steps: 474.20
# Mean for  Centralized / 2 : 477.08, std.dev.: 8.33, Distance: 95.86, Steps: 1000.00
# Mean for  Centralized / 4 : 401.16, std.dev.: 12.38, Distance: 98.11, Steps: 1000.00
# Overall Mean for  Centralized : 272.25, std.dev.: 163.49; CoT: 8.63; Vel.: 0.09
# TVel  && 272.25 & (163.49) && 0.08865434254786941 & (0.02) & 8.63
# 
