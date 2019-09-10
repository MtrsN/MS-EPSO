
import numpy as np

from .init_methods import Initialization

class MSEPSO:

    """MS-EPSO algorithm solver

    Required parameters
    -------------------
    fx : callable
        Aliase of the objective function to be optimized
    
    lb : numpy.array
        Lower bounds of the problem for each variable
    
    ub : numpy.array
        Upper bounds of the problem for each variable
    
    Optinal parameters
    ------------------
    tau : float [0, ~]
        Mutation rate used to mutate replica's weights (Default: 0.8)
    
    cp : float [0, 1]
        Communication probability used to access the best particle in the swarm (Default: 0.9)
    
    nr : int [1, ~]
        Number of generated replicas for each particle (Default: 1)
    
    mll : int [1, ~]
        The maximum local limit: number of evaluations to perform the particle-wise statistics with Gaussian distribution search (Default: 50)
    
    num_sol : int [5, ~]
        The number of particles to use in the optimization process (Default: 50)

    max_fes : int [num_sol, ~]
        Stopping criteria: maximum number of function evaluations (Default: 500). Note: iterations = max_fes/num_sol
    
    init_method : str
        Initialization method from 'init_methods' to generate candidate solutions (Default: 'uniform')
    
    seed : int
        Random state to guarantee reproducibility (Default: 101)
    """

    def __init__(self, **kwargs):

        required_parameters = ["fx", "lb", "ub"]
        optinal_parameters = ["tau", "cp", "nr", "mll", "max_fes", "num_sol", "init_method", "seed"]
        optinal_values = [0.8, 0.9, 1, 50, 500, 25, "uniform", 101]

        for k in kwargs:
            if(k not in required_parameters):
                raise TypeError('Missing required parameter for the algorithm: ' + str(k))
            if(k not in required_parameters and k not in optinal_parameters):
                raise TypeError('Unexpected keyword argument passed to the algorithm: ' + str(k))
        
        self.__dict__.update(kwargs)

        for op, ov in zip(optinal_parameters, optinal_values):
            if(not hasattr(self, op)):
                setattr(self, op, ov)

        # PSO
        self.particles = []
        self.particles_fitness = []
        self.particles_local_best = []
        self.particles_local_best_fitness = []
        self.best_particle = []
        self.best_particle_fitness = np.nan

        self.max_velocity = []
        self.min_velocity = []
        self.particles_velocity = []
        
        # EPSO
        self.all_weights = 4
        self.particles_weights = []

        # MS-EPSO
        self.particles_std = []
        self.particles_mean = []
        self.particles_local_limit = []
        self.particles_exp = []
    
    def initalize(self):
        """ Initialization phase of MS-EPSO"""
        self.solutions = Initialization(self.init_method).method(self.ns, self.lb, self.ub)


    def run(self):
        raise NotImplementedError
                
msepso = MSEPSO    