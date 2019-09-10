
from ms_epso import msepso
from problems import go_ampgo

ms_parameters = {}

if __name__ == "__main__":

    fx, gx, lb, ub = go_ampgo("Rosenbrock", dim= 30)

    ms = msepso(fx= fx, lb= lb, ub= ub, **ms_parameters)

    


    
    
