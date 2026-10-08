
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl 


#mpl.use('pgf')


WIDTH = 14 
HEIGHT = 3.5 

plt.rcParams['font.size'] = 14 
plt.rcParams['legend.fontsize'] = 14 

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(WIDTH, HEIGHT), gridspec_kw={'wspace':0.2, 'hspace':0.0}, frameon=False)


timesteps = np.arange(100, 12100, 100)
error_timesteps = np.arange(1000, 13000, 1000)



NUM_ITERATIONS = 20
TIME_DATA_POINT = 120

TIMESTEPS = np.arange(100, 12100, 100)
error_timesteps = np.arange(1000, 13000, 1000)
error_index = np.arange(9, 129, 10)

def get_rewards_vals(filename):

    rewards = pd.read_csv(filename)
    runs_rewards = [] 

    for i in range(NUM_ITERATIONS):
        runs_rewards.append(rewards.iloc[0 :TIME_DATA_POINT,0].to_numpy(dtype=np.float32))
        rewards.drop(index=rewards.index[0:TIME_DATA_POINT+1], inplace=True)

    standard_deviation = np.std(np.transpose(np.array(runs_rewards)), axis=1)
    runs_rewards = np.average(np.transpose(np.array(runs_rewards)), axis=1)

    standard_deviation = np.take(standard_deviation, error_index)

    error_locations = np.take(runs_rewards, error_index)

    return runs_rewards, error_locations, standard_deviation



deeptop_charging_rewards, deeptop_charging_error_locations, deeptop_charging_std = get_rewards_vals(f'mdp_results/output_DeepTOP_charging.txt')

deeptop_inventory_rewards, deeptop_inventory_error_locations, deeptop_inventory_std  = get_rewards_vals(f'mdp_results/output_DeepTOP_inventory.txt')


ddpg_charging_rewards, ddpg_charging_error_locations, ddpg_charging_std = get_rewards_vals(f'mdp_results/output_DDPG_charging.txt')

ddpg_inventory_rewards, ddpg_inventory_error_locations, ddpg_inventory_std  = get_rewards_vals(f'mdp_results/output_DDPG_inventory.txt')


td3_charging_rewards, td3_charging_error_locations, td3_charging_std = get_rewards_vals(f'mdp_results/output_td3_charging.txt')

td3_inventory_rewards, td3_inventory_error_locations, td3_inventory_std  = get_rewards_vals(f'mdp_results/output_td3_inventory.txt')

deadline_charging_rewards, deadline_charging_error_locations, deadline_charging_std  = get_rewards_vals(f'mdp_results/output_Deadline_Index_charging.txt')


deeptop_makestock_rewards, deeptop_makestock_error_locations, deeptop_makestock_std = get_rewards_vals(f'mdp_results/output_deepTOP_make_to_stock.txt')
ddpg_makestock_rewards, ddpg_makestock_error_locations, ddpg_makestock_std = get_rewards_vals(f'mdp_results/output_DDPG_make_to_stock.txt')
td3_makestock_rewards, td3_makestock_error_locations, td3_makestock_std = get_rewards_vals(f'mdp_results/output_td3_make_to_stock.txt')

salmut_makestock_rewards, salmut_makestock_error_locations, salmut_makestock_std = get_rewards_vals(f'mdp_results/output_salmut_make_to_stock.txt')

axes[0].plot(timesteps, deeptop_charging_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
axes[0].plot(timesteps, ddpg_charging_rewards, label='DDPG', zorder=4, color='C1', linestyle='dotted')
axes[0].plot(timesteps, td3_charging_rewards, label='TD3', zorder=3, color='C2', linestyle='dashdot')
deadline, = axes[0].plot(timesteps, deadline_charging_rewards, label='Deadline Index', zorder=2, color='C3', linestyle='dashed')


axes[0].errorbar(error_timesteps, deeptop_charging_error_locations, deeptop_charging_std, color='C0', alpha=0.6, capsize=2,zorder=1, ls='none')
axes[0].errorbar(error_timesteps, ddpg_charging_error_locations, ddpg_charging_std, color='C1', alpha=0.6, capsize=2,zorder=4, ls='none')
axes[0].errorbar(error_timesteps, td3_charging_error_locations, td3_charging_std, color='C2', alpha=0.6, capsize=2,zorder=3, ls='none')
axes[0].errorbar(error_timesteps, deadline_charging_error_locations, deadline_charging_std, color='C3', alpha=0.6, capsize=2,zorder=1, ls='none')




axes[1].plot(timesteps, deeptop_inventory_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
axes[1].plot(timesteps, ddpg_inventory_rewards, label='DDPG', zorder=4, color='C1', linestyle='dotted')
axes[1].plot(timesteps, td3_inventory_rewards, label='TD3', zorder=3, color='C2', linestyle='dashdot')


axes[1].errorbar(error_timesteps, deeptop_inventory_error_locations, deeptop_inventory_std, color='C0', alpha=0.6, capsize=2,zorder=1, ls='none')
axes[1].errorbar(error_timesteps, ddpg_inventory_error_locations, ddpg_inventory_std, color='C1', alpha=0.6, capsize=2,zorder=4, ls='none')
axes[1].errorbar(error_timesteps, td3_inventory_error_locations, td3_inventory_std, color='C2', alpha=0.6, capsize=2,zorder=3, ls='none')



deeptop, = axes[2].plot(timesteps, deeptop_makestock_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
ddpg, = axes[2].plot(timesteps, ddpg_makestock_rewards, label='DDPG', zorder=4, color='C1', linestyle='dotted')

td3, = axes[2].plot(timesteps, td3_makestock_rewards, label='TD3', zorder=3, color='C2', linestyle='dashdot')
salmut, = axes[2].plot(timesteps, salmut_makestock_rewards, label='SALMUT', zorder=2, color='C4', linestyle=(0, (3,1,3,3,1,3)))


axes[2].errorbar(error_timesteps, deeptop_makestock_error_locations, deeptop_makestock_std, color='C0', alpha=0.6, capsize=2,zorder=1, ls='none')
axes[2].errorbar(error_timesteps, ddpg_makestock_error_locations, ddpg_makestock_std, color='C1', alpha=0.6, capsize=2,zorder=4, ls='none')
axes[2].errorbar(error_timesteps, td3_makestock_error_locations, td3_makestock_std, color='C2', alpha=0.6, capsize=2,zorder=3, ls='none')
axes[2].errorbar(error_timesteps, salmut_makestock_error_locations, salmut_makestock_std, color='C4', alpha=0.6, capsize=2,zorder=2, ls='none')




yStart, yEnd = axes[0].get_ylim()

yLimits = np.linspace(yStart, yEnd, 9)
yTicks = [0.1*round(num/0.1) for num in yLimits]
axes[0].set_yticks(yTicks)


yStart, yEnd = axes[1].get_ylim()

yLimits = np.linspace(yStart, yEnd, 7)
yTicks = [10*round(num/10) for num in yLimits]
axes[1].set_yticks(yTicks)


axes[0].set_ylabel('Average Reward')

#axes[0].set_xlabel('Timesteps')
axes[1].set_xlabel('Timesteps')


xticks_vals = np.arange(0,13000, 4000)
axes[0].set_xticks(xticks_vals)
axes[1].set_xticks(xticks_vals)
axes[2].set_xticks(xticks_vals)


handles, labels = axes[0].get_legend_handles_labels()
#axes[0].xaxis.set_label_coords(0.5,-0.085)
axes[1].xaxis.set_label_coords(0.5,-0.085)


axes[2].legend(handles = [deeptop, ddpg, td3, deadline, salmut], 
    labels=['DeepTOP', 'DDPG', 'TD3', 'Deadline Index', 'SALMUT'],
    loc='lower right', bbox_to_anchor=(0.5, 1.02) , borderaxespad=0., ncol=5, frameon=False)


plt.savefig('mdp_results.pdf')
plt.show()