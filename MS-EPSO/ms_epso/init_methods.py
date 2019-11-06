
import numpy as np

class Initialization:

    """Base class for all initialization methods
    
    Parameters for all functions
    ----------------------------
    ns : int
        Number of solutions to initiliaze

    lb : np.array
        Lower bounds of the problem for each variable

    ub : np.array
        Upper bounds of the problem for each variable

    Return
    ------
    pop : 2D np.array
        Initialized population with 'ns' solutions  
    """

    def __init__(self, method_name):

        method_name = method_name.lower()

        methods = [strr.split("_")[1] for strr in dir(self) if strr.startswith("init_")]

        assert method_name in methods, "%s is not an available strategy. Select one from: %s" % (method_name, methods)

        self.method = getattr(self, "init_" + method_name)
    
    def init_uniform(self, ns, lb, ub):
        """Standard random uniform initialization"""
        return np.random.uniform(lb, ub, (ns, ub.shape[0]))