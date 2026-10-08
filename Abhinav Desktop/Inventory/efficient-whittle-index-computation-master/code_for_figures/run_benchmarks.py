"""
Run the benchmarks and save them in the file 'benchmarks/data'
"""
#! pip install markovianbandit-pkg
import markovianbandit as bandit
import os
import time
import datetime, platform

def time_on_random_bandit(dim, check_indexability, number_of_updates=0):
    model = bandit.random_restless(dim)
    ts = time.time()
    model.whittle_indices(check_indexability=check_indexability, number_of_updates=number_of_updates)
    return time.time()-ts

print("Warning: this should about 7h (or less, depending on the machine)\n")
print("we should let this code run and copy it into benchmarks/data/")

FILENAME = 'benchmarks/benchmarks_kka.csv'
if not(os.path.exists(FILENAME)):
    with open(FILENAME, 'w') as f:
        f.write('N,check_indexability,number_of_updates,run_time,timestamp,platform.version\n')


N = [10, 100, 500]
for i in range(1, 16):
    N.append(i*1000)
print('N    & check & check-update & no check & no-check update\\\\')
for dim in N:
    print('dim=',dim, flush=True, end='')
    for check_indexability in [True, False]:
        for number_of_updates in [0, '2n**0.1']:
            runtime = time_on_random_bandit(dim, check_indexability=check_indexability, number_of_updates=number_of_updates)
            if dim >= 1000:
                with open(FILENAME, 'a') as f:
                    line='{},{},{},{},{},{}\n'.format(dim, int(check_indexability),number_of_updates,runtime,
                        datetime.datetime.now().timestamp(),platform.version())
                    f.write(line)
                print('{:7.2f}'.format(runtime), end=' ', flush=True)
    print()
