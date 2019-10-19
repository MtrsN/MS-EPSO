
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from datetime import datetime
from ms_epso import msepso
from problems import go_lsgo

# Large Scale GLobal Optimization (LSGO) competition configs
runs = 25

ms_parameters = {
    "tau" : 0.8,
    "cp" : 0.9,
    "nr" : 1,
    "mll" : 50,
    "max_fes" : 3000000, # Also LSGO parameter
    "num_sol" : 100,
    "init_method" : "uniform",
}

def run_lsgo(function_id):

    print("Running function %d" % function_id, end= "\n\n")

    seeds = np.arange(101, 101 + runs)

    for seed in seeds:

        param_copy = cp.deepcopy(ms_parameters)
        param_copy.update({"seed" : seed})
        
        fx, _, lb, ub = go_lsgo(function_id)

        lb = np.array(lb)
        ub = np.array(ub)

        ms = msepso(fx= fx, lb= lb, ub= ub, **param_copy)
        ms.run()
    
if __name__ == "__main__":
    
    function_ids = np.arange(1, 16)
    
    num_threads = mp.cpu_count()
    
    p = mp.Pool(num_threads)

    p.map(run_lsgo, function_ids)

    p.close()

    p.join()