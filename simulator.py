import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from solver import Grid

class Simulator:
    """class for simulating"""

    def __init__(self, grid, steps_per_frame = 100):
        self.grid = grid
        self.steps_per_frame = steps_per_frame

        self.grid.ugrid[0][0][0] = 0
        self.grid.ugrid[0][self.grid.num_dx-1][0] = 5

        # Initialize figure for animation
        self.fig = plt.figure()
        self.ax = plt.subplot(xlim=(0, 50), ylim=(0, 50))
        self.im = self.ax.imshow(self.grid.ugrid[0], animated=True)
        self.fig.colorbar(self.im)


    def snapshot(self, step) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """

        if step > 0:
            print(step)
            self.grid.fwdEulerStep(step)
        self.im.set_array(self.grid.ugrid[step])
        return self.im,


    def animate(self):
        # nn = self.grid.num_timesteps
        nn = 2
        print('numframes: ' + str(nn))
        anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=3, blit=True, repeat=False)
        plt.show()  # show the animation


def main():
    grid = Grid()
    grid.initializeGrid()
    simulator = Simulator(grid=grid)
    simulator.animate()

if __name__ == "__main__":
    main()
