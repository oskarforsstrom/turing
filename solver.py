import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
# import simulator as Sim

class Grid:
    """class for grid"""

    def __init__(self,
    func = "Sch",
    num_timesteps = 20000,
    dt = 0.00008, # =< (dx^2 + dy^2)/(8*D_i) = 0.0005
    dx = 0.4,
    dy = 0.4,
    num_dx = 100,
    num_dy = 100,
    D_u = 1,
    D_v = 40,
    k=1, 
    c_=1,
    c1=0.1, 
    c2=0.9, 
    c3=1,
    c4=0,
    c5=0,
    no_flux=True,
        ):

        self.func = func
        
        self.num_timesteps = num_timesteps      # number of time steps - 1
        self.dt = dt                            # length of time step

        self.num_dx = num_dx                    # number of steps in x direction - 1
        self.num_dy = num_dy                    # number of steps in y direction - 1
        self.dx = dx                            # length of step in x direction
        self.dy = dy                            # length of step in y direction

        self.D_u = D_u                          # Diffusion rate for u
        self.D_v = D_v                          # Diffusion rate for v
        self.k = k                              # Reaction parameters
        self.c_ = c_
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5

        self.no_flux = no_flux

        self.ugrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))
        self.vgrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))


    def fwdEulerStep(self, n):
        h = self.dx # = dy
        k = self.dt
        u = self.ugrid
        v = self.vgrid
        D_u = self.D_u
        D_v = self.D_v
        f = getattr(self, self.func + '_f')
        g = getattr(self, self.func + '_g')

        # u 
        for x in range(1,self.num_dx-1):
            for y in range(1,self.num_dy-1):
                u[n+1][x][y] = ( u[n][x][y] + ((D_u*k) / h**2) * 
                            (u[n][x+1][y] + u[n][x-1][y] + u[n][x][y+1] + u[n][x][y-1] - 4*u[n][x][y]) + k*f(u[n][x][y], v[n][x][y]) )
                v[n+1][x][y] = ( v[n][x][y] + ((D_v*k) / h**2) * 
                            (v[n][x+1][y] + v[n][x-1][y] + v[n][x][y+1] + v[n][x][y-1] - 4*v[n][x][y]) + k*g(u[n][x][y], v[n][x][y]) )

        if self.no_flux:

            for x in range(1, self.num_dx-1):
                u[n+1][x][0] = u[n+1][x][1]
                u[n+1][x][self.num_dy - 1] = u[n+1][x][self.num_dy - 2]

                v[n+1][x][0] = v[n][x][1]
                v[n+1][x][self.num_dy - 1] = v[n+1][x][self.num_dy - 2]


            for y in range(1, self.num_dy-1):
                u[n+1][0][y] = u[n+1][1][y]
                u[n+1][self.num_dx - 1][y] = u[n+1][self.num_dx - 2][y]

                v[n+1][0][y] = v[n+1][1][y]
                v[n+1][self.num_dx - 1][y] = v[n+1][self.num_dx - 2][y]


    def integrate(self):
        for n in range(self.num_timesteps-1):
            self.fwdEulerStep(n)

    # generate homogenous grid with random perturbations
    def initializeGrid(self):
        u_star = (1/self.k)*(self.c1 + self.c2)
        v_star = (self.c3/self.c2) * (1/u_star**2)
        ones = np.ones((self.num_dx, self.num_dy))

        self.ugrid[0] = (u_star)*ones + np.random.uniform(low=-0.2, high=0.2, size=(self.num_dx, self.num_dy))
        self.vgrid[0] = (v_star)*ones + np.random.uniform(low=-0.2, high=0.2, size=(self.num_dx, self.num_dy))

    
    # Grierer-Meinhardt reaction functions
    def GM_f(self, u, v):
        return self.c1 - self.c2*u + self.c3*( u**2 / ((1 + self.k*u**2)*v) )

    def GM_g(self, u, v):
        return self.c4*u**2 - self.c5*v

    # Schnakenberg reaction functions
    def Sch_f(self, u, v):
        return self.c1 - self.c_*u + self.c3*u**2*v     

    def Sch_g(self, u, v):
        return self.c2 - self.c3*u**2*v

    
        

def main():
    
    grid = Grid(func = "Sch")
    grid.animate()


if __name__ == "__main__":
    main()
