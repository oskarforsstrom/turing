import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from solver import Grid
import os

class Simulator:
    """class for simulating"""

    def __init__(self, grid, steps_per_frame = 100, colormin=0.0, colormax=2):
        self.grid = grid
        self.steps_per_frame = steps_per_frame
        print(self.grid.ugrid[0])

        bootgrid = ((colormin+colormax)/2)*np.ones((self.grid.num_dx, self.grid.num_dy))
        bootgrid[0][0] = colormin
        bootgrid[-1][-1] = colormax

        # Initialize figure for animation
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.im = self.ax.imshow(bootgrid, animated=True, cmap="turbo")
        self.fig.colorbar(self.im)


    def snapshot(self, step) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """

        if step > 0:
            step = step - 1
            if step%10==0:
                print('t: ' + str(step*self.grid.dt))
            self.grid.fwdEulerStep(step)
            
        self.im.set_array(self.grid.ugrid[step])
        return self.im,


    def animate(self):
        nn = self.grid.num_timesteps
        print('numframes: ' + str(nn))
        anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=1, blit=True, repeat=False)
        plt.show()  # show the animation
