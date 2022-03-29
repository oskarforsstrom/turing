from solverspectral import Grid

import os
import numpy as np
from datetime import datetime

import multiprocessing as mp

def run_var_param(param, varied_parameter, func, tend):
    """worker function for varying parameters on simulations"""

    print('var par: ' + varied_parameter + ', par: ' + str(param) +', START')
    grid = Grid(**{varied_parameter: param}, func=func)
    if not grid.param_check():
        print("NÄJ")
    grid.initializeGrid()
    wresult = grid.integrate(tend=tend)
    print(wresult)
    print('var par: ' + varied_parameter + ', par: ' + str(param) +', STOP')
    return (param, wresult)

def app_var_param():
    """
    Common: D_u, D_v
    
    Sch: c_, c_1, c_2, c_3

    GM: c_1, c_2, c_3, c_4, c_5, k
    """

    func = 'Sch'
    tend = 1
    varied_parameter = 'c_'
    var_mid = 1
    
    var_range = 0.1
    var_num = 2

    params = np.linspace(-var_range, var_range, var_num)
    params = params + var_mid

    arg_combos = [[param, varied_parameter, func, tend] for param in params]

    nprocs = mp.cpu_count()
    print('nprocs: ' + str(nprocs) + '\n')
    pool = mp.Pool(processes=nprocs)
    result = pool.starmap(run_var_param, arg_combos)

    param_save = [t[0] for t in result]
    w_save =[t[1] for t in result]

    dir = './var_param_spectral/'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Create path : {}'.format(dir))
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

    if not os.path.exists(dir + dt_string + '/'):
        os.makedirs(dir + dt_string + '/')
        print('Create path : {}'.format(dir + dt_string + '/'))

    param_path = dir + dt_string + '/param'
    w_path = dir + dt_string + '/w'

    np.save(param_path, np.array(param_save))
    np.save(w_path, np.array(w_save))

def main():
    app_var_param()

if __name__ == '__main__':
    main()