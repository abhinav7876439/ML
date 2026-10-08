

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl 

#mpl.use('pgf')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


WIDTH = 14 
HEIGHT = 3.5

plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12 

plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(WIDTH, HEIGHT),  gridspec_kw={'wspace':0.2, 'hspace':0.0}, frameon=False)


ARMS = [10, 20, 30]
BUDGET = [3, 5, 6]

NUM_ITERATIONS = 20
TIME_DATA_POINT = 120

TIMESTEPS = np.arange(100, 12100, 100)
error_timesteps = np.arange(1000, 13000, 1000)
error_index = np.arange(9, 129, 10)

def get_rewards_vals(filename):

    runs_rewards = [] 

    for i in range(NUM_ITERATIONS):
        
        rewards = pd.read_csv(filename+f"_{i+1}.txt")
        runs_rewards.append(rewards.iloc[0 :TIME_DATA_POINT,0].to_numpy(dtype=np.float32))


    standard_deviation = np.std(np.transpose(np.array(runs_rewards)), axis=1)
    runs_rewards = np.average(np.transpose(np.array(runs_rewards)), axis=1)

    standard_deviation = np.take(standard_deviation, error_index)

    error_locations = np.take(runs_rewards, error_index)

    return runs_rewards, error_locations, standard_deviation



deeptop_eight_rewards, deeptop_eight_error_locations, deeptop_eight_std  = get_rewards_vals(f'recovering_RMAB/results/output_DeepTOP_A{ARMS[0]}B{BUDGET[0]}')
deeptop_ten_rewards, deeptop_ten_error_locations, deeptop_ten_std  = get_rewards_vals(f'recovering_RMAB/results/output_DeepTOP_A{ARMS[1]}B{BUDGET[1]}')
deeptop_thirty_rewards, deeptop_thirty_error_locations, deeptop_thirty_std  = get_rewards_vals(f'recovering_RMAB/results/output_DeepTOP_A{ARMS[2]}B{BUDGET[2]}')


neural_lpql_eight_rewards, neural_lpql_eight_error_locations, neural_lpql_eight_std  = get_rewards_vals(f'recovering_RMAB/results/output_LPQL_A{ARMS[0]}B{BUDGET[0]}')
neural_lpql_ten_rewards, neural_lpql_ten_error_locations, neural_lpql_ten_std  = get_rewards_vals(f'recovering_RMAB/results/output_LPQL_A{ARMS[1]}B{BUDGET[1]}')
neural_lpql_thirty_rewards, neural_lpql_thirty_error_locations, neural_lpql_thirty_std  = get_rewards_vals(f'recovering_RMAB/results/output_LPQL_A{ARMS[2]}B{BUDGET[2]}')


neural_wibql_eight_rewards, neural_wibql_eight_error_locations, neural_wibql_eight_std  = get_rewards_vals(f'recovering_RMAB/results/output_WIBQL_A{ARMS[0]}B{BUDGET[0]}')
neural_wibql_ten_rewards, neural_wibql_ten_error_locations, neural_wibql_ten_std  = get_rewards_vals(f'recovering_RMAB/results/output_WIBQL_A{ARMS[1]}B{BUDGET[1]}')
neural_wibql_thirty_rewards, neural_wibql_thirty_error_locations, neural_wibql_thirty_std  = get_rewards_vals(f'recovering_RMAB/results/output_WIBQL_A{ARMS[2]}B{BUDGET[2]}')


tabular_lpql_eight_rewards, tabular_lpql_eight_error_locations, tabular_lpql_eight_std  = get_rewards_vals(f'tabular_methods/tabular_results/tabular_lpql_arms_{ARMS[0]}_budget_{BUDGET[0]}_run')
tabular_lpql_ten_rewards, tabular_lpql_ten_error_locations, tabular_lpql_ten_std  = get_rewards_vals(f'tabular_methods/tabular_results/tabular_lpql_arms_{ARMS[1]}_budget_{BUDGET[1]}_run')
tabular_lpql_thirty_rewards, tabular_lpql_thirty_error_locations, tabular_lpql_thirty_std  = get_rewards_vals(f'tabular_methods/tabular_results/tabular_lpql_arms_{ARMS[2]}_budget_{BUDGET[2]}_run')


tabular_wibql_eight_rewards, tabular_wibql_eight_error_locations, tabular_wibql_eight_std  = get_rewards_vals(f'recovering_tabular_methods/tabular_results/tabular_wibql_arms_{ARMS[0]}_budget_{BUDGET[0]}_run')
tabular_wibql_ten_rewards, tabular_wibql_ten_error_locations, tabular_wibql_ten_std  = get_rewards_vals(f'recovering_tabular_methods/tabular_results/tabular_wibql_arms_{ARMS[1]}_budget_{BUDGET[1]}_run')
tabular_wibql_thirty_rewards, tabular_wibql_thirty_error_locations, tabular_wibql_thirty_std  = get_rewards_vals(f'recovering_tabular_methods/tabular_results/tabular_wibql_arms_{ARMS[2]}_budget_{BUDGET[2]}_run')

neurwin_ten_rewards, neurwin_ten_error_locations, neurwin_ten_std = get_rewards_vals(f'recovering_RMAB/results/output_neurwin_A{ARMS[0]}B{BUDGET[0]}')
neurwin_twenty_rewards, neurwin_twenty_error_locations, neurwin_twenty_std = get_rewards_vals(f'recovering_RMAB/results/output_neurwin_A{ARMS[1]}B{BUDGET[1]}')
neurwin_thirty_rewards, neurwin_thirty_error_locations, neurwin_thirty_std = get_rewards_vals(f'recovering_RMAB/results/output_neurwin_A{ARMS[2]}B{BUDGET[2]}')



axes[0].plot(TIMESTEPS, deeptop_eight_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
axes[0].errorbar(error_timesteps, deeptop_eight_error_locations, deeptop_eight_std, color='C0', alpha=0.6, capsize=2, zorder=1, ls='none')


axes[0].plot(TIMESTEPS, neural_lpql_eight_rewards, label='Neural LPQL', zorder=2, color='C1', linestyle='dotted')
axes[0].errorbar(error_timesteps, neural_lpql_eight_error_locations, neural_lpql_eight_std, color='C1', alpha=0.6, capsize=2, zorder=1, ls='none')


axes[0].plot(TIMESTEPS, neural_wibql_eight_rewards, label='Neural WIBQL', zorder=3, color='C2', linestyle=(0, (3,1,1,1,1,1)))
axes[0].errorbar(error_timesteps, neural_wibql_eight_error_locations, neural_wibql_eight_std, color='C2', alpha=0.6, capsize=2, zorder=1, ls='none')


axes[0].plot(TIMESTEPS, tabular_lpql_eight_rewards, label='Tabular LPQL', zorder=4, color='C3', linestyle='dashdot')
axes[0].errorbar(error_timesteps, tabular_lpql_eight_error_locations, tabular_lpql_eight_std, color='C3', alpha=0.6, capsize=2, zorder=4, ls='none')

axes[0].plot(TIMESTEPS, tabular_wibql_eight_rewards, label='Tabular WIBQL', zorder=5, color='C4', linestyle=(0, (3,1,3,3,1,3)))
axes[0].errorbar(error_timesteps, tabular_wibql_eight_error_locations, tabular_wibql_eight_std, color='C4', alpha=0.6, capsize=2, zorder=5, ls='none')

axes[0].plot(TIMESTEPS, neurwin_ten_rewards, label='NeurWIN', zorder=6, color='C5', linestyle= (0, (1, 1)))
axes[0].errorbar(error_timesteps, neurwin_ten_error_locations, neurwin_ten_std, color='C5', alpha=0.6, capsize=2, zorder=6, ls='none')


axes[1].plot(TIMESTEPS, deeptop_ten_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
axes[1].errorbar(error_timesteps, deeptop_ten_error_locations, deeptop_ten_std, color='C0', alpha=0.6, capsize=2, zorder=1, ls='none')

axes[1].plot(TIMESTEPS, neural_lpql_ten_rewards, label='Neural LPQL', zorder=2, color='C1', linestyle='dotted')
axes[1].errorbar(error_timesteps, neural_lpql_ten_error_locations, neural_lpql_ten_std, color='C1', alpha=0.6, capsize=2, zorder=1, ls='none')


axes[1].plot(TIMESTEPS, neural_wibql_ten_rewards, label='Neural WIBQL', zorder=3, color='C2', linestyle=(0, (3,1,1,1,1,1)))
axes[1].errorbar(error_timesteps, neural_wibql_ten_error_locations, neural_wibql_ten_std, color='C2', alpha=0.6, capsize=2, zorder=1, ls='none')


axes[1].plot(TIMESTEPS, tabular_lpql_ten_rewards, label='Tabular LPQL', zorder=4, color='C3', linestyle='dashdot')
axes[1].errorbar(error_timesteps, tabular_lpql_ten_error_locations, tabular_lpql_ten_std, color='C3', alpha=0.6, capsize=2, zorder=4, ls='none')

axes[1].plot(TIMESTEPS, tabular_wibql_ten_rewards, label='Tabular WIBQL', zorder=5, color='C4', linestyle=(0, (3,1,3,3,1,3)))
axes[1].errorbar(error_timesteps, tabular_wibql_ten_error_locations, tabular_wibql_ten_std, color='C4', alpha=0.6, capsize=2, zorder=5, ls='none')

axes[1].plot(TIMESTEPS, neurwin_twenty_rewards, label='NeurWIN', zorder=6, color='C5', linestyle= (0, (1, 1)))
axes[1].errorbar(error_timesteps, neurwin_twenty_error_locations, neurwin_twenty_std, color='C5', alpha=0.6, capsize=2, zorder=6, ls='none')



axes[2].plot(TIMESTEPS, deeptop_thirty_rewards, label='DeepTOP', zorder=1, color='C0', linestyle='solid')
axes[2].errorbar(error_timesteps, deeptop_thirty_error_locations, deeptop_thirty_std, color='C0', alpha=0.6, capsize=2, zorder=1, ls='none')

axes[2].plot(TIMESTEPS, neural_lpql_thirty_rewards, label='Neural LPQL', zorder=2, color='C1', linestyle='dotted')
axes[2].errorbar(error_timesteps, neural_lpql_thirty_error_locations, neural_lpql_thirty_std, color='C1', alpha=0.6, capsize=2, zorder=2, ls='none')


axes[2].plot(TIMESTEPS, neural_wibql_thirty_rewards, label='Neural WIBQL', zorder=3, color='C2', linestyle=(0, (3,1,1,1,1,1)))
axes[2].errorbar(error_timesteps, neural_wibql_thirty_error_locations, neural_wibql_thirty_std, color='C2', alpha=0.6, capsize=2, zorder=3, ls='none')


axes[2].plot(TIMESTEPS, tabular_lpql_thirty_rewards, label='Tabular LPQL', zorder=4, color='C3', linestyle='dashdot')
axes[2].errorbar(error_timesteps, tabular_lpql_thirty_error_locations, tabular_lpql_thirty_std, color='C3', alpha=0.6, capsize=2, zorder=4, ls='none')

axes[2].plot(TIMESTEPS, tabular_wibql_thirty_rewards, label='Tabular WIBQL', zorder=5, color='C4', linestyle=(0, (3,1,3,3,1,3)))
axes[2].errorbar(error_timesteps, tabular_wibql_thirty_error_locations, tabular_wibql_thirty_std, color='C4', alpha=0.6, capsize=2, zorder=5, ls='none')


axes[2].plot(TIMESTEPS, neurwin_thirty_rewards, label='NeurWIN', zorder=6, color='C5', linestyle= (0, (1, 1)))
axes[2].errorbar(error_timesteps, neurwin_thirty_error_locations, neurwin_thirty_std, color='C5', alpha=0.6, capsize=2, zorder=6, ls='none')


yStart, yEnd = axes[0].get_ylim()

yLimits = np.linspace(yStart, yEnd, 6)
yTicks = [1*round(num/1) for num in yLimits]
axes[0].set_yticks(yTicks)


yStart, yEnd = axes[1].get_ylim()

yLimits = np.linspace(yStart, yEnd, 10)
yTicks = [1*round(num/1) for num in yLimits]
axes[1].set_yticks(yTicks)


yStart, yEnd = axes[2].get_ylim()

yLimits = np.linspace(yStart, yEnd, 10)
yTicks = [1*round(num/1) for num in yLimits]
axes[2].set_yticks(yTicks)

handles, labels = axes[2].get_legend_handles_labels()

xticks_vals = np.arange(0,13000, 3000)
axes[0].set_xticks(xticks_vals)
axes[1].set_xticks(xticks_vals)
axes[2].set_xticks(xticks_vals)

axes[0].legend(loc='lower right', bbox_to_anchor=(3.27, 1.02) , borderaxespad=0., ncol=6, frameon=False)

axes[0].set_ylabel('Average Reward')
axes[1].set_xlabel('Timesteps')
axes[1].xaxis.set_label_coords(0.5,-0.08)


plt.savefig(f'recovering_rmabs_results_arms_{ARMS}_budgets_{BUDGET}.pdf')

plt.show()



