from solver import Grid
from simulator import Simulator

import os
import numpy as np

def main():
    dx=0.4
    dy=0.4
    dt=0.0001
    D_u = 1
    grid = Grid(no_flux=False, periodic=True, D_u=D_u, dt=dt)
    grid.initializeGrid()
    if not grid.param_check():
        print("NÄJ")
    simulator = Simulator(grid=grid, colormin=0.5, colormax=1.5)
    simulator.animate()

    pattern_list = os.listdir('./patterns/')

    # Find relevant pattern number to add in pattern filename
    if len(pattern_list) != 0:
        for pattern in pattern_list:
            if grid.func not in pattern or "v" in pattern:
                pattern_list.remove(pattern)
        N = int(sorted(pattern_list)[-1][-5]) + 1
    else:
        N = 1

    save_time = 1000

    if grid.ugrid[save_time].any(): # Checks that the grid at the specified time is not zero. np.ndarray.any() == False if it's the zero matrix
        u_file = "./patterns/{}_u_{}".format(grid.func, N) # ex: Sch_u_2, GM_u_15
        v_file = "./patterns/{}_v_{}".format(grid.func, N)
        
        np.save(u_file, grid.ugrid[save_time])
        np.save(v_file, grid.vgrid[save_time])

if __name__ == '__main__':
    main()