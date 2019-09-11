# MS-EPSO

Implementation of the Maximum Search Limitations: Evolutionary Particle Swarm Optimization.

## Usage

    In 'main.py':
        1) Set the objective function (the current example is for standard optimization benchmarks);
        2) Set the parameters in the dictionary;
        3) Run 'main.py'

## FAQ

    1) How to install "benchmarks"?
        1.1) Go to SciPy - https://github.com/scipy/scipy
        1.2) Get SciPy/benchmarks
        1.3) Paste it in your site-package folder
            1.3.1) In Windows it is usually in '/Python/Lib/site-packages'
            1.3.2) You can get the folder by doing the following in command line:
                1.3.2.1) python
                1.3.2.2) import numpy as np; print(np.__file__)
