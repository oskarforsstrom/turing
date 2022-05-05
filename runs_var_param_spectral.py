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
    
    Sch: c_, c1, c2, c3

    GM: c1, c2, c3, c4, c5, k
    """

    func = 'Sch'
    tend = 20
    varied_parameter = 'c_'
    var_mid = 0.8
    
    var_range = 0.2
    var_num = 15

    params = np.linspace(-var_range, var_range, var_num)
    params = params + var_mid

    arg_combos = [[param, varied_parameter, func, tend] for param in params]

    nprocs = mp.cpu_count()
    print('nprocs: ' + str(nprocs) + '\n')
    pool = mp.Pool(processes=nprocs)
    result = pool.starmap(run_var_param, arg_combos)

    param_save = [t[0] for t in result]
    w_save =[t[1] for t in result]
    grid = Grid()
    numdxdy_save = [grid.num_dx, grid.num_dy]

    dir = './var_param_spectral/'
    
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Create path : {}'.format(dir))
    
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    parametercode = '_' + varied_parameter
    if not os.path.exists(dir + dt_string + parametercode +'/'):
        os.makedirs(dir + dt_string + parametercode + '/')
        print('Create path : {}'.format(dir + dt_string + parametercode + '/'))

    param_path = dir + dt_string + parametercode + '/param'
    w_path = dir + dt_string + parametercode + '/w'
    numdxdy_path = dir + dt_string + parametercode + '/numdxdy'

    np.save(param_path, np.array(param_save))
    np.save(w_path, np.array(w_save))
    np.save(numdxdy_path, np.array(numdxdy_save))



def get_param_range(param, func):

    # initialize grid to be able to check its parameters
    grid = Grid(func=func)
    orig_param = param

    delta = 0.01
    vals = []

    # loop over large value range for given parameter and adds
    # it to vals
    for value in range(10000):
        setattr(grid, param, value*delta) # set grid parameter to current parameter value

        try: # handle divisions by zero
            check = grid.param_check() # if parameters permit TP
            if check:
                vals.append(value) # saves integer
        except:
            continue

    if len(vals) != 0:
        idx = 0
        ranges = [[vals[0]]] # ranges[idx] is the idx'th range for the parameter
        prev = vals[0]

        # Appends all values to ranges. If there are gaps in vals, they are added to separate lists in ranges
        for val in vals[1:]:
            if val - prev > 1:
                ranges.append([val])
                idx += 1
            else: # no gap between last and current
                ranges[idx].append(val)
            
            prev = val

        intervals = []

        for rng in ranges:
            if len(rng) > 1:
                lo = delta*rng[0]
                hi = delta*rng[-1]
                intervals.append([lo, hi])
            else:
                lo = delta*rng[0]
                hi = lo
                intervals.append([lo,hi])

        setattr(grid, param, orig_param) 
        return intervals
    else:
        print("No range found for {} for given parameters.".format(param))
        setattr(grid, param, orig_param)
        return None



def main():
    app_var_param()

if __name__ == '__main__':
    main()