import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from solver import Grid
import os

class Simulator:
    """class for simulating"""

    def __init__(self, grid, steps_per_frame = 100):
        self.grid = grid
        self.steps_per_frame = steps_per_frame
        print(self.grid.ugrid[0])

        self.grid.ugrid[0][0][0] = 0.8
        self.grid.ugrid[0][self.grid.num_dx-1][0] = 1.2

        # Initialize figure for animation
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.im = self.ax.imshow(self.grid.ugrid[0], animated=True, cmap="turbo")
        self.fig.colorbar(self.im)


    def snapshot(self, step) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """

        if step > 0:
            step = step - 1
            if step%10==0:
                print(step)
            self.grid.fwdEulerStep(step)
            
        self.im.set_array(self.grid.ugrid[step])
        return self.im,


    def animate(self):
        nn = self.grid.num_timesteps
        print('numframes: ' + str(nn))
        anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=1, blit=True, repeat=False)
        plt.show()  # show the animation


def main():
    grid = Grid()
    grid.initializeGrid()
    simulator = Simulator(grid=grid)
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
    u_file = "./patterns/{}_u_{}".format(grid.func, N) # ex: Sch_u_2, GM_u_15
    v_file = "./patterns/{}_v_{}".format(grid.func, N)

    np.save(u_file, grid.ugrid[save_time])
    np.save(v_file, grid.vgrid[save_time])


if __name__ == "__main__":
    main()
