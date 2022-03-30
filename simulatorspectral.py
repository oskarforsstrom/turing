import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from solverspectral import Grid
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


    def snapshot(self, step, w0, frames_per_t) :

        if step > 0:
            step = step - 1
            if step%10==0:
                print('t: ' + str(step*self.grid.dt))


            wresult = self.grid.integrate_step(w0=w0, frames_per_t=frames_per_t) 
            w = wresult[-1]
            w = self.grid.arraytogrids(w)
            self.grid.ugrid[step] = w[0]
            self.grid.vgrid[step] = w[1]
            w0 = w

        self.im.set_array(self.grid.ugrid[step]) ################
        return self.im,


    def animate(self, frames_per_t):
        w0 = np.array([self.grid.ugrid[0],self.grid.vgrid[0]]) # initialize w0

        nn = self.grid.num_timesteps
        print('numframes: ' + str(nn))
        anim = animation.FuncAnimation(self.fig, self.snapshot, fargs=(w0, frames_per_t),
            frames=nn, interval=1, blit=True, repeat=False)
        plt.show()  # show the animation


    def simulate(self, tend, plotInit=False):
        # does tend no. of spectral time steps and converts back to u and v
        wresult = self.grid.integrate(tend)
        w = wresult[-1]
        w = self.grid.arraytogrids(w)
        u = w[0]
        v = w[1]

        if plotInit:
            # plot u
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('u', fontsize=20)
            ax1.set_title("initial state")
            ax2.set_title("final state")
            im = ax1.imshow(self.grid.vgrid[0])
            ax2.imshow(u)
            fig.colorbar(im)
            plt.show()

            # plot v
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('v', fontsize=20)
            ax1.set_title("initial state")
            ax2.set_title("final state")
            im = ax1.imshow(self.grid.vgrid[0])
            ax2.imshow(u)
            fig.colorbar(im)
            plt.show()
        else:
            # plot u
            fig = plt.figure()
            fig.suptitle("u", fontsize=20)
            ax = plt.subplot()
            im = ax.imshow(u)
            fig.colorbar(im)
            plt.show()

            # plot v
            fig = plt.figure()
            fig.suptitle("v", fontsize=20)
            ax = plt.subplot()
            im = ax.imshow(v)
            fig.colorbar(im)
            plt.show()

    # simulates multiple runs with same initial grid
    def initStateGrind(self, num_runs, tend, morphogen):

        runs = []

        # do multiple runs with same initial grid
        self.grid.initializeGrid()
        init_u = self.grid.ugrid[0]
        init_v = self.grid.vgrid[0]
        for i in range(num_runs):
            wresult = self.grid.integrate(tend=tend)
            w = wresult[-1]
            w = self.grid.arraytogrids(w)
            runs.append(w)

            # reset grid after one run
            self.grid.ugrid = np.zeros((self.grid.num_timesteps, self.grid.num_dx, self.grid.num_dy))
            self.grid.vgrid = np.zeros((self.grid.num_timesteps, self.grid.num_dx, self.grid.num_dy))
            self.grid.ugrid[0] = init_u
            self.grid.vgrid[0] = init_v

        if morphogen == "u":
            state = 0
        else:
            state = 1

        for run in runs:
            morph = run[state]
            fig = plt.figure()
            fig.suptitle(morphogen, fontsize=20)
            ax = plt.subplot()
            im = ax.imshow(morph)
            fig.colorbar(im)
            plt.show()

        
