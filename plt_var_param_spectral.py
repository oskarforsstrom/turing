from solverspectral import Grid

import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    dir = './var_param_spectral/'

    if len(sys.argv) > 1:
        seq_id = sys.argv[1]
        
        varied_parameter = ''
        for idx, char in enumerate(reversed(seq_id)):
            if char == '_' and idx != 0:
                break
            else:
                varied_parameter += char

        varied_parameter = ''.join(list(reversed(varied_parameter)))

    else:

        """--------------------"""
        """Lägg in manuellt här"""
        """--------------------"""

        dt_string = '04-04-2022_23-20-00'
        varied_parameter = 'c_' 
        parametercode = '_' + varied_parameter

        seq_id = dt_string + parametercode

    load_dir = dir + seq_id

    param = np.load(load_dir + '/' + 'param.npy')
    w = np.load(load_dir + '/' + 'w.npy')
    numdxdy = np.load(load_dir + '/' + 'numdxdy.npy')
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
        im1 = ax1.imshow(u)
        im2 = ax2.imshow(v)
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.show()


if __name__ == '__main__':
    main()