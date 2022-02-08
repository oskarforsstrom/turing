import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class Grid:
    """class for grid"""

    def __init__(self,
    func = "Sch",
    steps_per_frame = 100,
    num_timesteps = 1000,
    dt = 0.1,
    dx = 0.1,
    dy = 0.1,
    num_dx = 100,
    num_dy = 100,
    D_u = 1,
    D_v = 40,
    k=1, 
    c1=0.1, 
    c2=0.9, 
    c3=1,
    c4=0,
    c5=0,
    noflux=True,
        ):

        self.func = func
        
        self.steps_per_frame = steps_per_frame      # number of integration steps per frame
        self.num_timesteps = num_timesteps      # number of time steps - 1
        self.dt = dt                            # length of time step

        self.num_dx = num_dx                    # number of steps in x direction - 1
        self.num_dy = num_dy                    # number of steps in y direction - 1
        self.dx = dx                            # length of step in x direction
        self.dy = dy                            # length of step in y direction

        self.D_u = D_u                          # Diffusion rate for u
        self.D_v = D_v                          # Diffusion rate for v
        self.k = k                              # Reaction parameters
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.noflux = noflux

        self.ugrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))
        self.vgrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))

        # Initialize figure for animation
        self.fig = plt.figure()
        self.ax = plt.subplot(xlim=(0, self.num_dx), ylim=(0, self.num_dy))

    def fwdEulerStep(self, n):
        h = self.dx # = dy
        k = self.dt
        u = self.ugrid
        v = self.vgrid
        D_u = self.D_u
        f = getattr(self, self.func + '_f')
        g = getattr(self, self.func + '_g')

        # u 
        for x in range(1,self.num_dx-1):
            for y in range(1,self.num_dy-1):
                u[n+1][x][y] = ( u[n][x][y] + ((D_u*k) / h**2) * 
                            (u[n][x+1][y] + u[n][x-1][y] + u[n][x][y+1] + u[n][x][y-1] - 4*u[n][x][y]) + k*f(u[n][x][y], v[n][x][y]) )

        if self.no_flux:

            for x in range(1, self.num_dx-1):
                u[n][x][0] = u[n][x][1]
                u[n][x][self.num_dy] = u[n][x][self.num_dy - 1]

            for y in range(1, self.num_dy-1):
                u[n][0][y] = u[n][1][y]
                u[n][self.num_dx][y] = u[n][self.num_dx - 1][y]


    def integrate(self):
        for n in range(self.num_timesteps-1):
            self.fwdEulerStep(n)


    def initializeGrid(self):
        # generate homogenous grid with random perturbations
        init_grid = np.ones((self.num_dx, self.num_dy)) + np.random.uniform(low=-0.1, high=0.1, size=(self.num_dx-1, self.num_dy-1))
        self.ugrid[0] = init_grid


    def snapshot(self) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """

        self.integrate()
        return self.ax.scatter(self.x, self.y, s=1500, marker='o', c="r"),


    def animate(self):

        nn = self.num_timesteps//self.stepsPerFrame 
        anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=50, blit=True, repeat=False)
        plt.show()  # show the animation

    
    # Grierer-Meinhardt reaction functions
    def GM_f(self, u, v):
        return self.c1 - self.c2*u + self.c3*( u**2 / ((1 + self.k*u**2)*v) )

    def GM_g(self, u, v):
        return self.c4*u**2 - self.c5*v

    # Schnakenberg reaction functions
    def Sch_f(self, u, v):
        return self.c1 - self.k*u + self.c3*u**2*v     # k substitutes c_-1 in the paper

    def Sch_g(self, u, v):
        return self.c2 - self.c3*u**2*v
        

def main():
    
    grid = Grid(func = "Sch")
    grid.animate()


if __name__ == "__main__":
    main()
