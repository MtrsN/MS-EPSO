
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
        Initialization method from 'init_methods' to generate candidate particles (Default: 'uniform')
    
    seed : int
        Random state to guarantee reproducibility (Default: 101)
    """

    def __init__(self, **kwargs):

        required_parameters = ["fx", "lb", "ub"]
        optinal_parameters = ["tau", "cp", "nr", "mll", "max_fes", "num_sol", "init_method", "seed"]
        optinal_values = [0.8, 0.9, 1, 50, 500, 25, "uniform", 101]

        assert all(req in list(kwargs.keys()) for req in required_parameters), "Missing required parameters. Required parameters: %s" % required_parameters

        for k in kwargs:
            if(k not in required_parameters and k not in optinal_parameters):
                raise TypeError('Unexpected keyword argument passed to the algorithm: ' + str(k))
        
        
        self.__dict__.update(kwargs)

        for op, ov in zip(optinal_parameters, optinal_values):
            if(not hasattr(self, op)):
                setattr(self, op, ov)
        
        self.dim = self.ub.shape[0]

        # PSO
        self.particles = []
        self.particles_fitness = []
        self.particles_local_best = []
        self.particles_local_best_fitness = []
        self.best_particle = []
        self.best_particle_fitness = float(10e9)

        self.max_velocity = []
        self.min_velocity = []
        self.particles_velocity = []
        
        # EPSO
        self.total_weights = 4
        self.particles_weights = []

        # MS-EPSO
        self.particles_std = []
        self.particles_mean = []
        self.particles_limit = []
        self.particles_exp = []

        # Best solution tracking
        self.history = []

        self.fes = int(0)
    
    def initalize(self):
        """ Initialization phase of MS-EPSO"""

        np.random.seed(101)

        # PSO
        self.particles = Initialization(self.init_method).method(self.num_sol, self.lb, self.ub)
        self.particles_fitness = np.fromiter((self.fx(sol) for sol in self.particles), self.particles.dtype, self.particles.shape[0])
        self.particles_local_best = np.copy(self.particles)
        self.particles_local_best_fitness = np.copy(self.particles_fitness)

        self.max_velocity = np.abs(self.ub - self.lb)
        self.min_velocity = self.max_velocity * -1
        self.particles_velocity = np.random.uniform(self.min_velocity, self.max_velocity, (self.num_sol, self.dim))

        # EPSO
        self.particles_weights = np.random.uniform(size= (self.num_sol, self.total_weights))

        # MS-EPSO
        self.particles_std = np.std(self.particles, axis= 1)
        self.particles_mean = np.mean(self.particles, axis= 1)
        self.particles_limit = np.zeros(self.particles.shape[0])
        self.particles_exp = np.ones(self.particles.shape[0])

        self.fes = int(self.num_sol)

        self.update_best()

    def update_best(self):

        best = np.argmin(self.particles_local_best_fitness)

        if(self.particles_local_best_fitness[best] < self.best_particle_fitness):
            self.best_particle = np.copy(self.particles_local_best[best])
            self.best_particle_fitness = self.particles_local_best_fitness[best]

        self.history.append(self.best_particle_fitness)

    def compare(self, solution_1, solution_2):
        """ Compare two tuples and returns the one with best fitness value.

        Note: It assumes that the fitness value is in the second position of the tuple"""

        s1_fitness = solution_1[1]
        s2_fitness = solution_2[1]

        if(s1_fitness < s2_fitness):
            return solution_1
        else:
            return solution_2

    def evaluate(self, x):
        """Increment the fes value, evaluate the solution and compare it against the global best"""

        self.fes += 1
        
        f_x = self.fx(x)

        if(f_x < self.best_particle_fitness):
            self.best_particle_fitness = f_x
            self.best_particle = np.copy(x)
        
        if(self.fes % self.num_sol == 0):
            self.history.append(self.best_particle_fitness)
        
        return f_x

    def enforce(self, x, lb, ub):
        """Set the solution between the limits of the search space"""
        upper_violation = x > ub
        lower_violation = x < lb

        x[upper_violation] = ub[upper_violation]
        x[lower_violation] = lb[lower_violation]

        return x

    def update_velocity(self, x, v, xhat, x_w):
        """ Updates the velocity of a solution based on EPSO rule"""

        communications = np.random.uniform(size= self.dim) < self.cp

        inertia = x_w[0] * v
        memory = x_w[1] * (xhat - x)
        cooperation = x_w[2] * ((self.best_particle * (1 + x_w[3] * np.random.normal(size= self.dim))) - x)

        new_velocity = inertia + memory + (cooperation * communications)

        return self.enforce(new_velocity, self.min_velocity, self.max_velocity)

    def update_position(self, x, v):
        """Update the particle solution according to PSO/EPSO rules"""
        new_position = x + v

        return self.enforce(new_position, self.lb, self.ub)

    def generate_gauss_particle(self, mu, sigma):
        """ Produces a new particle with specific mean and standard deviation using gaussian distribution"""
        sol = np.random.normal(mu, sigma, self.dim)
        sol = self.enforce(sol, self.lb, self.ub)

        sol_fx = self.evaluate(sol)

        if(self.shouldEnd()):
            return (True, )
        else:
            return (sol, sol_fx)

    def generate_epso_replicas(self, x, v, xhat, x_w):
        """Generate NR replicas based on EPSO approach"""

        best_replica_weights = self.mutate_weights(x_w)
        best_replica_velocity = self.update_velocity(x, v, xhat, best_replica_weights)
        best_replica_position = self.update_position(x, best_replica_velocity)
        best_replica_fitness = self.evaluate(best_replica_position)

        if(self.shouldEnd()):
            return (True, )

        for _ in range(self.nr - 1):

            new_replica_weights = self.mutate_weights(x_w)
            new_replica_velocity = self.update_velocity(x, v, xhat, new_replica_weights)
            new_replica_position = self.update_position(x, new_replica_velocity)
            new_replica_fitness = self.evaluate(new_replica_position)

            new_tuple = (new_replica_position, new_replica_fitness, new_replica_violations, new_replica_velocity, new_replica_weights)
            best_tuple = (best_replica_position, best_replica_fitness, best_replica_violations, best_replica_velocity, best_replica_weights)

            (best_replica_position, best_replica_fitness,
            best_replica_violations, best_replica_velocity, best_replica_weights) = self.compare(new_tuple, best_tuple)

            if(self.shouldEnd()):
                return (True, )

        return best_replica_position, best_replica_fitness, best_replica_velocity, best_replica_weights

    def move_particle(self, x, v, xhat, x_w):
        
        new_particle_velocity = self.update_velocity(x, v, xhat, x_w)
        new_particle = self.update_position(x, new_particle_velocity)
        new_particle_fitness = self.evaluate(new_particle)

        if(self.shouldEnd()):
            return (True, )
        else:
            return new_particle, new_particle_fitness, v, x_w

    def mutate_weights(self, weights):
        """Weight mutation according to EPSO rule"""

        mutations = np.random.normal(size= self.total_weights) * self.tau

        mutated_weights = weights + mutations

        mutated_weights[mutated_weights > 1] = 1
        mutated_weights[mutated_weights < 0] = 0

        return mutated_weights

    def shouldEnd(self):
        if(self.fes >= self.max_fes):
            return True
        else:
            return False

    def run(self):
        """"Optimization process of MS-EPSO
        
        Note: if the tuple size is 1 during the optimization process, it means that 
        the stopping criteria was reached while generating a specific solution"""

        self.initalize()

        while(not self.shouldEnd()):

            for i in range(self.num_sol):

                # In all rules, replicas are generated using EPSO mechanism.

                #  Particle generated with Gaussian distribution with best particle statistics (Particle first)
                rule1 = self.particles_limit[i] < self.mll and self.particles_exp[i]

                # EPSO approach (Replicas first > standard particle movement)
                rule2 = self.particles_limit[i] < self.mll and not self.particles_exp[i]

                # Particle generated with Gaussian distribution with local best particle statistics (Particle first)
                rule3 = self.particles_limit[i] >= self.mll

                if(rule1):

                    # Gaussian particle
                    mu = np.mean(self.best_particle)
                    sigma = np.std(self.best_particle)

                    new_particle = self.generate_gauss_particle(mu, sigma)
                    new_particle += (self.particles_velocity[i], self.particles_weights[i], )

                    if(len(new_particle) == 1): return True

                    # EPSO replicas moving from the gaussian particle
                    best_replica = self.generate_epso_replicas(new_particle[0], self.particles_velocity[i], self.particles_local_best[i], self.particles_weights[i])

                    if(len(best_replica) == 1): return True
                
                if(rule2):
                    
                    # EPSO replicas moving from i-th particle
                    best_replica = self.generate_epso_replicas(self.particles[i], self.particles_velocity[i], self.particles_local_best[i], self.particles_weights[i])
                    
                    if(len(best_replica) == 1): return True

                    # EPSO particles
                    new_particle = self.move_particle(self.particles[i], self.particles_velocity[i], self.particles_local_best[i], self.particles_weights[i])

                    if(len(new_particle) == 1): return True

                if(rule3):

                    # Gaussian particle
                    mu = self.particles_mean[i]
                    sigma = self.particles_std[i]

                    new_particle = self.generate_gauss_particle(mu, sigma)
                    new_particle += (self.particles_velocity[i], self.particles_weights[i], )

                    if(len(new_particle) == 1): return True

                    # EPSO replicas moving from the gaussian particle
                    best_replica = self.generate_epso_replicas(new_particle[0], self.particles_velocity[i], self.particles_local_best[i], self.particles_weights[i])

                    if(len(best_replica) == 1): return True

                    self.particles_limit[i] = 0

                    if(self.particles_exp[i]):
                        self.particles_exp[i] = 0
                
                # Tournament 1: Best replica vs New Particle -> x_i^{g+1}
                (self.particles[i], self.particles_fitness[i], self.particles_velocity[i], self.particles_weights[i]) = self.compare(best_replica, new_particle)

                particle_tuple = (self.particles[i], self.particles_fitness[i])
                particle_best_tuple = (self.particles_local_best[i], self.particles_local_best_fitness[i])

                # Tournament 2: \hat{x}_i^{g+1} vs x_i^{g+1}
                (self.particles_local_best[i], self.particles_local_best_fitness[i]) = self.compare(particle_tuple, particle_best_tuple)

                # If there is a new local best for the i-th particle, save its particle-wise statistics, else, increment the limit
                if(particle_tuple[1] == self.particles_local_best_fitness[i]):
                    self.particles_std[i] = np.std(self.particles[i])
                    self.particles_mean[i] = np.mean(self.particles[i])
                else:
                    self.particles_limit[i] += 1

# Aliase      
msepso = MSEPSO