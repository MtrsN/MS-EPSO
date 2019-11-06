
import numpy as np

def go_ampgo(func_name= "Rosenbrock", dim= 30):

	"""
	Objective function initialization.

	Parameters
	----------
	func_name : string
		Name of the desired objective function from Scipy - Go Benchmark Functions

	dim : int
		Desired dimensionality for the problem.

	Returns
	-------
	fun :	function
		The function which evaluates the problem
	
	vio : None
		Generalization for experiment file.

	N	:	int
		Dimensionality of the problem

	max :	sequence
		Upper bounds for the problem

	min :	sequence
		Lower bounds for the problem
	"""

	import benchmarks.benchmarks.go_benchmark_functions as gbf

	func = getattr(gbf, func_name)(dimensions= dim)

	real_func = validate_function(func, func_name)

	return real_func.fun, None, real_func.xmin, real_func.xmax

def validate_function(func, func_name):

	import benchmarks.benchmarks.go_benchmark_functions as gbf
	
	real_dim = len(func.global_optimum[0])

	func_aliase = getattr(gbf, func_name)

	if(func.N != real_dim):

		if(real_dim == 0):
			return func_aliase()
		else:
			return func_aliase(dimensions= real_dim)
	
	return func