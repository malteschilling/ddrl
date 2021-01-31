# DDRL
Deep Decentralized Reinforcement Learning - locally structured architecture for DRL in a continuous locomotion control task.


## Results

### Experiment 1 -- DRL for Variation of the Controller Architecture for the Four-Legged Agent

Fig. 7 - Learning curves for different controller architectures over learning time on flat terrain, Mean return per episode---calculated over 10 seeds---for all different learning architectures is shown during learning, given in simulation steps (interactions with the environment).

visualize_learning_performance.py

Fig. 8 - Learning performance over learning time, evaluated as the average score over time during the learning phase. It is evaluated as the mean of the episode return over learning epochs during that particular run which is proportional to the area under the learning curves.

Evaluation:
compare_generalization_trained_on_flat.py

Visualization:
visualize_generalization_different_heightfields_pd.py

### Evaluation of Trained Controller --- Generalization to novel terrain

Fig. 9 - Evaluation on different types of terrain: mean return for four selected architectures is shown and how performance decreases for more difficult terrain. 

visualize_generalization_different_heightfields_pd.py

### Experiment 2 -- Variation of Neural Network Models

Fig. 10 - Comparison of locomotion performance for different sizes of neural networks used in the architectures. Mean performance (y axis) over ten seeds for each neural network configuration. Horizontal axis represents size of networks.

visualize_nn_size_variations_pd.py

### Experiment 3 - Learning to walk at a given target speed in uneven terrain

Fig. 11 - Learning curves for different controller architectures over learning time, trained to reach two target velocities on increasingly more uneven terrain.

visualize_learning_over_time_tvel.py

Fig. 12 - Evaluation on different types of terrain: For each architecture, all trained controller are evaluated.

visualize_generalization_different_heightfields_tvel_pd.py

### Evaluation for variation of velocity

Fig. 13, Fig. 14 - Evaluation of different target velocities on uneven terrain: For each architecture, all trained controller are evaluated for $10$ runs at different target velocities (x axis, ranging from $0.5$ to $2.5$ in increments of $0.1$).

visualize_generalization_variationVelocity_pd.py

### Analysis of Importance of Input Features

Fig. 15 - Importance map showing how a small change in one input (rows, y-axis, named on the left) feature dimension affects the control signals (columns, shown on the bottom). 

visualize_evaluated_grads_centralized.py