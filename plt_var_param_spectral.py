from solverspectral import Grid

import numpy as np
import matplotlib.pyplot as plt

def main():
    dir = './var_param_spectral/'
    
    dt_string = '30-03-2022_16-27-13'
    varied_parameter = 'c1' 
    parametercode = '_' + varied_parameter

    param = np.load(dir + dt_string + parametercode + '/' + 'param.npy')
    w = np.load(dir + dt_string + parametercode + '/' + 'w.npy')
    numdxdy = np.load(dir + dt_string + parametercode + '/' + 'numdxdy.npy')
    grid = Grid(num_dx = numdxdy[0], num_dy=numdxdy[1])
    for idx, param in enumerate(param):
        wtemp = w[idx][-1]
        wtemp = grid.arraytogrids(wtemp)
        u = wtemp[0]
        v  = wtemp[1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(varied_parameter + ' = ' + str(param))
        ax1.set_title("u")
        ax2.set_title("v")
        im = ax1.imshow(u)
        ax2.imshow(v)
        fig.colorbar(im)
        plt.show()


if __name__ == '__main__':
    main()