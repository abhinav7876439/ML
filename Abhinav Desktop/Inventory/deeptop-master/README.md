This code is for the NeurIPS 2022 paper:


Khaled Nakhleh, I-Hong Hou. DeepTOP: Deep Threshold-Optimal Policy for MDPs and RMABs. In Advances in Neural Information Processing Systems (NeurIPS 2022), volume 36, December 2022.


## Description
---


The algorithm was implemented for the Markov Decision Process (MDP), and the Restless Multi-Armed Bandits (RMABs) settings.
Corresponding settings are stored in separate directories (recovering bandits setting is under the "recovering_bandits_rmab" directory).

As an example to run an MDP algorithm, run from the MDP directory:
```
python3 -u main_DeepTOP_charging.py > output_DeepTOP_charging.txt &
python3 -u main_DeepTOP_inventory.py > output_DeepTOP_inventory.txt &
python3 -u main_DeepTOP_make_to_stock.py > output_DeepTOP_make_to_stock.txt &
```

Also for the RMAB algorithm run: 

```
python3 -u main_DeepTOP.py --nb_arms 10 --budget 3 > output_DeepTOP_A10B3.txt & 
python3 -u main_DeepTOP.py --nb_arms 20 --budget 5 > output_DeepTOP_A20B5.txt & 
python3 -u main_DeepTOP.py --nb_arms 30 --budget 6 > output_DeepTOP_A30B6.txt & 
```

## Acknowledgment
---

The source code relies on classes and functions from other open-source repositories. 
Cited code includes recognition in its respective file.

The LPQL and WIBQL implementations under "tabular_methods/" and "recovering_bandits_rmab/recovering_tabular_methods/"
were taken from the repository: https://github.com/killian-34/MAIQL_and_LPQL 

