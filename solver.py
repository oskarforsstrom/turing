import numpy as np

class Grid:
    """class for grid"""

    def __init__(self,
    num_timesteps = 1000,
    dt = 0.1,
    dx = 0.1,
    dy = 0.1,
    num_dx = 100,
    num_dy = 100,
        ):
        self.num_timesteps = num_timesteps      # number of time steps - 1
        self.dt = dt                            # length of time step

        self.num_dx = num_dx                    # number of steps in x direction - 1
        self.num_dy = num_dy                    # number of steps in y direction - 1
        self.dx = dx                            # length of step in x direction
        self.dy = dy                            # length of step in y direction

        self.ugrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))
        self.vgrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))

# Grierer-Meinhardt reaction functions
def GM_f(u,v, k=0, c1=0, c2=0, c3=0):
    return c1 - c2*u + c3*( u**2 / ((1 + k*u**2)*v) )

def GM_g(u,v, c4=0, c5=0):
    return c4*u**2 -c5*v

