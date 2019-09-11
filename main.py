
import numpy as np
import matplotlib.pyplot as plt
import timeit

from ms_epso import msepso
from problems import go_ampgo

from math import floor, ceil, exp, log

ms_parameters = {
    "tau" : 0.8,
    "cp" : 0.9,
    "nr" : 1,
    "mll" : 25,
    "max_fes" : 100000,
    "num_sol" : 50,
    "init_method" : "uniform",
    "seed" : 101
}

def round_power_of_10(n, f= "min"):

    exp = log(n, 10)
    
    if(f is "min"):
        exp = floor(exp)
    else:
        exp = ceil(exp)
    
    return 10 ** exp

def build_plot(history):

    x_name = "# Function Evaluations"
    y_name = "Fitness"

    fig, ax = plt.subplots(1, 1, figsize= (9, 8))

    ax.set_xscale("symlog")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    plot_range = (np.arange(0, len(history)) + 1) * ms_parameters["num_sol"]

    ax.semilogy(plot_range, history, label= "MS-EPSO", color= "red", marker= "*", markevery= 0.1, markersize= 10, linewidth= 3)
    
    plt.gca().yaxis.grid(True, color= "black", linewidth= 0.1)
    plt.gca().xaxis.grid(True, color= "black", linewidth= 0.1)
        
    start, end = ax.get_ylim()
        
    ax.set_ylim(bottom= round_power_of_10(start, "min"), top= round_power_of_10(end, "max"))

    plt.show()

if __name__ == "__main__":

    fx, gx, lb, ub = go_ampgo("Rosenbrock", dim= 30)

    ms = msepso(fx= fx, lb= lb, ub= ub, **ms_parameters)

    start = timeit.default_timer()

    ms.run()

    end = timeit.default_timer()

    print("Processing time: %ss" % (end - start))

    build_plot(ms.history)