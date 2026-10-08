import sys
sys.path.append('')
from tests.test_nonregression import compute_for_random_example
import numpy as np
import time

if __name__ == "__main__":
    N = [10, 100, 500]
    for i in range(1, 6): N.append(i*1000)
    numbers_of_updates =  [0, '2n**0.1']
    table_title = 'N    '
    for update in numbers_of_updates:
        table_title += '&{} no test&   {}     '.format(update, update)
    print(table_title, '\\\\')
    for dim in N:
        print('{:5d}'.format(dim), end='&', flush=True)
        for number_of_updates in numbers_of_updates:
            for check_indexability in [False, True]:
                print("{:8.2f}".format(compute_for_random_example(dim,
                    check_indexability=check_indexability, number_of_updates=number_of_updates, return_time=True)),
                    end=' &', flush=True)
        print('\\\\')
