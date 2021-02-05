import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def boxplot_annotate_brackets_group(num1, target_list, data, center, height, dh=.0, barh=20, fs=None, maxasterix=None, offset=0):
    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05
        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'
    num_lines = 0
    ry = max(height)
    lx, ly = center[num1], height[num1]
    max_y = max(ly, ry)
    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh_int = dh*(ax_y1 - ax_y0)
    for num2 in target_list:
        rx = center[num2]
        #barh *= (ax_y1 - ax_y0)
        y = max_y + dh_int + 150 + 60*num_lines + offset
        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        #barx = [lx, rx]
        #bary = [y, y]
        mid = ((lx+rx)/2, y+35)
        plt.plot(barx, bary, c='black')
        print(barx, bary)
        kwargs = dict(ha='center', va='bottom')
        num_lines +=1
    if fs is not None:
        kwargs['fontsize'] = fs
    plt.text(*mid, text, **kwargs)
    
"""
    Visualizes the learning performances after 20M steps and does 
    statistics (non-parametric, unpaired) on the different groups.
    
    For Experiment 1 = trained on flat terrain, aiming for high velocity.
    
    Directly loads progress.csv log files from training (not provided in git repository).
    This shows a highly significant learning performance difference 
    (stats not provided in detail in paper, but see sect 4.1)
    
    Learning performance over learning time: evaluated as the average score over time 
    during the learning phase. It is important to distinguish this from the learning 
    curves which show the mean return at a specific point in time. 
    Learning performance was especially introduced as a measure for learning up to a 
    specific point in time. It is meant as a summary of all learning up to that point 
    in time. It is evaluated as the mean of the episode return over learning epochs 
    during that particular run which is proportional to the area under the learning curves.

"""
if __name__ == "__main__":
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
    
    # Load data - ! progress.csv are not given in git (are too huge)
    exp_path = [os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Centralized', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_FullyDecentral', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_Local', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_SingleDiagonal', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_SingleNeighbor', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_SingleToFront', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoDiags', 
         os.getcwd() + '/Results/experiment_1_models_architectures_on_flat/HF_10_QuantrupedMultiEnv_TwoSides'] 
     
    experiment_dirs = [[os.path.join(exp_path_item,dI) for dI in os.listdir(exp_path_item) if os.path.isdir(os.path.join(exp_path_item,dI))] for exp_path_item in exp_path]

    # Access progress.csv files and accumulate learning performance.
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
            
            mean_cum_rew_new = np.cumsum(rew_new)/np.arange(1,rew_new.shape[0]+1)
            if i==0:
                mean_cum_rew = np.vstack([mean_cum_rew_new])
                time_steps = (df.iloc[:,6].values)
            else:
                mean_cum_rew = np.vstack([mean_cum_rew,mean_cum_rew_new])
        rew_mean = np.mean(reward_values, axis=0)
        rew_std = np.std(reward_values, axis=0)
        rew_lower_std = rew_mean - rew_std
        rew_upper_std = rew_mean + rew_std
        all_exp_data.append( [rew_mean, rew_std, rew_lower_std, rew_upper_std, reward_values, mean_cum_rew] )
        print("Loaded ", exp_dir)

    # Look at different points in time (5M, 10M, 15M, 20M steps)
    architecture_samples_at312 = np.zeros((8,10))
    for arch in range(0, architecture_samples_at312.shape[0]):
        for i in range(0,10):
            architecture_samples_at312[arch][i] = all_exp_data[arch][4][i][311]
    architecture_samples_at625 = np.zeros((8,10))
    for arch in range(0, architecture_samples_at625.shape[0]):
        for i in range(0,10):
            architecture_samples_at625[arch][i] = all_exp_data[arch][4][i][624]
    architecture_samples_at1250 = np.zeros((8,10))
    for arch in range(0, architecture_samples_at1250.shape[0]):
        for i in range(0,10):
            architecture_samples_at1250[arch][i] = all_exp_data[arch][4][i][1249]
        
    architecture_learn_perf_samples_at1250 = np.zeros((8,10))
    for arch in range(0, architecture_learn_perf_samples_at1250.shape[0]):
        for i in range(0,10):
            architecture_learn_perf_samples_at1250[arch][i] = all_exp_data[arch][5][i][1249]
    
    #########################################
    # Statistics: Test for differences between groups
    #
    # We analyzed the learning performance of the unpaired samples from the different 
    # architectures using the non-parametric Kruskal-Wallis test [Kruskal & Wallis 1952] 
    # (as the data appears not normally distributed) and for post-hoc analysis using the 
    # Dunn [Dunn & Dunn 1961] post-hoc test (applying Bonferroni correction) 
    # following [Raff 2019].
    #########################################
    from scipy import stats
    import scikit_posthocs as sp
    stats.kruskal(architecture_samples_at312[0], architecture_samples_at312[1], 
        architecture_samples_at312[2], architecture_samples_at312[3], 
        architecture_samples_at312[4], architecture_samples_at312[5], 
        architecture_samples_at312[6], architecture_samples_at312[7])
    sp.posthoc_mannwhitney(architecture_samples_at312, p_adjust = 'holm')
    sp.posthoc_mannwhitney(architecture_samples_at312, p_adjust = 'bonferroni')
    # OR posthoc_dunn, posthoc_conover
    stats.kruskal(architecture_samples_at625[0], architecture_samples_at625[1], 
        architecture_samples_at625[2], architecture_samples_at625[3], 
        architecture_samples_at625[4], architecture_samples_at625[5], 
        architecture_samples_at625[6], architecture_samples_at625[7])
    sp.posthoc_mannwhitney(architecture_samples_at625, p_adjust = 'holm')
    sp.posthoc_mannwhitney(architecture_samples_at625, p_adjust = 'bonferroni')
    
    stats.kruskal(architecture_samples_at1250[0], architecture_samples_at1250[1], 
        architecture_samples_at1250[2], architecture_samples_at1250[3], 
        architecture_samples_at1250[4], architecture_samples_at1250[5], 
        architecture_samples_at1250[6], architecture_samples_at1250[7])
    sp.posthoc_mannwhitney(architecture_samples_at1250, p_adjust = 'holm')
    sp.posthoc_mannwhitney(architecture_samples_at1250, p_adjust = 'bonferroni')

    stats.kruskal(architecture_learn_perf_samples_at1250[0], architecture_learn_perf_samples_at1250[1],
        architecture_learn_perf_samples_at1250[2], architecture_learn_perf_samples_at1250[3],
        architecture_learn_perf_samples_at1250[4], architecture_learn_perf_samples_at1250[5],
        architecture_learn_perf_samples_at1250[6], architecture_learn_perf_samples_at1250[7])
    sp.posthoc_mannwhitney(architecture_learn_perf_samples_at1250, p_adjust = 'bonferroni')

    # Plotting: Box-plot
    ###########
    arch_names = [exp_path[i].split('_')[-1] for i in range(0,8)]
    colors = [tableau20[i] for i in range(0,16,2)]
    medianprops = dict(color="black",linewidth=1.5)
    # Plotting functions
    fig = plt.figure(figsize=(9, 6))
    # Remove the plot frame lines. They are unnecessary chartjunk.  
    ax_lperf = plt.subplot(111)  
    ax_lperf.spines["top"].set_visible(False)  
    ax_lperf.spines["right"].set_visible(False) 
    ax_lperf.set_ylabel('Learning Performance')
    #ax_lperf.violinplot(architecture_learn_perf_samples_at1250.transpose())
    bplot = ax_lperf.boxplot(architecture_learn_perf_samples_at1250.transpose(), patch_artist=True, medianprops=medianprops, sym='+')
    #ax_lperf.set_ylim(0, 3000)
    ax_lperf.set_xticklabels(arch_names,
                        rotation=45, fontsize=10)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    xpos_box = np.arange(1,9)
    heights_box = np.mean(architecture_learn_perf_samples_at1250, axis=1)
    boxplot_annotate_brackets_group(0, [6], 'p < 0.05', xpos_box, heights_box)
    boxplot_annotate_brackets_group(7, [6,3], 'p < 0.05', xpos_box, heights_box, offset=300)
    boxplot_annotate_brackets_group(0, [1,2,3,4,5,7], 'p < 0.01', xpos_box, heights_box, offset=500)
    fig.tight_layout()
    
    # KruskalResult(statistic=42.42148148148152, pvalue=4.313157988082117e-07)
    # eta2 = 0.464
    # From https://rpkgs.datanovia.com/rstatix/reference/kruskal_effsize.html
    # eta2[H] = (H - k + 1)/(n - k); where H is the value obtained in the Kruskal-Wallis test; k is the number of groups; n is the total number of observations.
    #The eta-squared estimate assumes values from 0 to 1 and multiplied by 100 indicates the percentage of variance in the dependent variable explained by the independent variable. The interpretation values commonly in published litterature are: 0.01- < 0.06 (small effect), 0.06 - < 0.14 (moderate effect) and >= 0.14 (large effect).
    # Rea & Parker (1992) their interpretation for r, the following:
    #0.00 < 0.01 - Negligible
    #0.01 < 0.04 - Weak
    #0.04 < 0.16 - Moderate
    #0.16 < 0.36 - Relatively strong
    #0.36 < 0.64 - Strong
    #0.64 < 1.00 - Very strong
    # Epsilon Squared = H / ((n^2-1)/(n+1)); 0.423

    # sp.posthoc_mannwhitney(architecture_learn_perf_samples_at1250, p_adjust = 'bonferroni')
    #           1         2         3         4         5         6         7         8
    # 1  1.000000  0.005115  0.005115  0.005115  0.005115  0.005115  0.012310  0.005115
    # 2  0.005115  1.000000  1.000000  1.000000  1.000000  1.000000  0.483209  0.255038
    # 3  0.005115  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000  0.101094
    # 4  0.005115  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000  0.036818
    # 5  0.005115  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000  0.255038
    # 6  0.005115  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000  0.591750
    # 7  0.012310  0.483209  1.000000  1.000000  1.000000  1.000000  1.000000  0.012310
    # 8  0.005115  0.255038  0.101094  0.036818  0.255038  0.591750  0.012310  1.000000
    
    # sp.posthoc_dunn(architecture_learn_perf_samples_at1250, p_adjust = 'bonferroni')
    #               1         2         3         4         5        6         7             8
    # 1  1.000000e+00  0.001752  0.009997  0.041893  0.004362  0.00137  1.000000  4.497767e-08
    # 2  1.751523e-03  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  1.000000e+00
    # 3  9.997414e-03  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  3.853953e-01
    # 4  4.189260e-02  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  1.194134e-01
    # 5  4.362285e-03  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  6.816147e-01
    # 6  1.369941e-03  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  1.000000e+00
    # 7  1.000000e+00  1.000000  1.000000  1.000000  1.000000  1.00000  1.000000  2.232285e-03
    # 8  4.497767e-08  1.000000  0.385395  0.119413  0.681615  1.00000  0.002232  1.000000e+00