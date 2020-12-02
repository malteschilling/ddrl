import numpy as np
import matplotlib.pyplot as plt

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

data_smoothn_steps = np.array([1., 0.9, 0.8, 0.7, 0.6])
# Data from generalization of architectures: architecture trained on flat terrain,
# evaluated on 8 different uneven terrain (see smoothness above, 1. = flat).
# 0 - centralized, 1 - fully dec, 2 - local, 
# 3 - singe diag, 4 - single neig.
# 5 - two contr. diag, 6 - two neighb. contr.
# 7 - connections towards front
data_arch = np.zeros((8,5))
data_arch[0,:] = [2611.26, 2376.81, 1363.48, 515.60, 301.56]
data_arch[1,:] = [2525.04, 2290.15,  556.51, 198.27,-342.04]
data_arch[2,:] = [2463.77, 2424.02, 1723.96, 810.73, 260.17]
data_arch[3,:] = [2526.31, 2135.60,  814.81, 293.08,-101.08]
data_arch[4,:] = [2405.77, 2116.85, 1296.74, 310.82,-274.03]
data_arch[5,:] = [2373.74, 1944.98, 1274.33, 160.35,  52.44]
data_arch[6,:] = [2608.21, 2145.66,  716.41, 431.02,  73.49]
data_arch[7,:] = [2412.72, 2282.36, 1388.30, 125.64,-319.63]
data_std = np.zeros((8,5))
data_std[0,:] = [743.62, 939.29, 1118.48, 885.11, 776.12]
data_std[1,:] = [410.93, 701.86,  865.05, 736.25, 417.28]
data_std[2,:] = [260.86, 406.89,  589.46, 825.33, 505.31]
data_std[3,:] = [463.73, 829.59, 1062.01, 752.74, 476.94]
data_std[4,:] = [419.15, 742.48,  717.43, 918.88, 336.04]
data_std[5,:] = [615.19, 766.56,  972.42, 679.97, 563.10]
data_std[6,:] = [928.88, 986.02,  898.93, 814.87, 561.16]
data_std[7,:] = [327.95, 521.30,  684.85, 706.68, 310.59]

data_min = data_arch - data_std
data_max = data_arch + data_std

exp_name = ['Centralized', 'Fully Decentralized', 'Local Neighbors', 
    'Single Diag. Neigh.', 'Single Neigh.', 
    'Two contr. diag.', 'Two contr. neighb.', 'Towards Front']

# Plotting functions
####################
fig = plt.figure(figsize=(6, 8))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False) 

#ax_arch.set_yscale('log')
ax_arch.set_xlim(1., 0.6)
#ax_arch.set_ylim(0, 800)  

for i in range(0, 3):
    # Use matplotlib's fill_between() call to create error bars.   
    plt.fill_between(data_smoothn_steps, data_min[i,:],  
                     data_max[i,:], color=tableau20[i*2 + 1], alpha=0.25)  
    plt.plot(data_smoothn_steps, data_arch[i,:], color=tableau20[i*2], lw=1, label=exp_name[i])

ax_arch.set_xlabel('Smoothness', fontsize=14)
ax_arch.set_ylabel('Return per Episode', fontsize=14)
plt.legend(loc="upper right")
#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')

# Box plot - DATA MUST BE SAVED IN EVALUATION IN OTHER FORMAT
fig = plt.figure(figsize=(10, 6))
# Remove the plot frame lines. They are unnecessary chartjunk.  
ax_arch = plt.subplot(111)  
ax_arch.spines["top"].set_visible(False)  
ax_arch.spines["right"].set_visible(False)   
ax_arch.set_xlim(1., 0.6)
for i in range(0, 8):
    # Use matplotlib's fill_between() call to create error bars.   
    plt.fill_between(data_smoothn_steps, data_min[i,:],  
                     data_max[i,:], color=tableau20[i*2 + 1], alpha=0.25)  
    plt.plot(data_smoothn_steps, data_arch[i,:], color=tableau20[i*2], lw=1, label=exp_name[i])

ax_arch.set_xlabel('Smoothness', fontsize=14)
ax_arch.set_ylabel('Return per Episode', fontsize=14)
plt.legend(loc="upper right")
#plt.plot([0,500], [200,200], color=tableau20[6], linestyle='--')


#plt.legend(loc="upper right")
plt.show()