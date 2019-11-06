
def go_lsgo(func_id):

    from cec2013lsgo.cec2013 import Benchmark
    
    bench = Benchmark()

    bench_info = bench.get_info(func_id)

    dim = bench_info['dimension']

    return bench.get_function(func_id), None, [bench_info['lower']] * dim, [bench_info['upper']] * dim