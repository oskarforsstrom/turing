import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import gaussian_filter
import math as m

from scipy.fft import fft2, ifft2
from scipy.integrate import odeint

class Grid:
    """class for grid"""

    def __init__(self,
    num_timesteps = 20000,
    dt = 0.00001, # =< (dx^2 + dy^2)/(8*D_i) = 0.0005
    dx = 0.4,
    dy = 0.4,
    num_dx = 50,
    num_dy = 50,
    D_u = 1,
    D_v = 40,
    k=1, 
    c_=1,
    c1=0.1, 
    c2=0.9, 
    c3=1,
    c4=0,
    c5=0,
        ):
        
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

        self.ugrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))
        self.vgrid = np.zeros((self.num_timesteps, self.num_dx, self.num_dy))

    # generate homogenous grid with random perturbations
    def initializeGrid(self):
        u_star = (1/self.k)*(self.c1 + self.c2) ## only schnaken
        v_star = (self.c3/self.c2) * (1/u_star**2) ## only schnaken
        ones = np.ones((self.num_dx, self.num_dy))

        self.ugrid[0] = (u_star)*ones + np.random.uniform(low=-0.05, high=0.05, size=(self.num_dx, self.num_dy))
        self.vgrid[0] = (v_star)*ones + np.random.uniform(low=-0.05, high=0.05, size=(self.num_dx, self.num_dy))

    def integrate(self, tend=2):
        self.initializeGrid()
        t = np.linspace(0,tend,tend+1)
        w0 = np.array([self.ugrid[0],self.vgrid[0]])
        w0 = self.gridstoarray(w0)
        wresult = odeint(self.rhsf, w0, t, printmessg=True)
        w = wresult[-1]
        w = self.arraytogrids(w)
        u = w[0]
        v = w[1]
        fig = plt.figure()
        ax = plt.subplot()
        im = ax.imshow(u)
        fig.colorbar(im)
        plt.show()

        fig = plt.figure()
        ax = plt.subplot()
        im = ax.imshow(v)
        fig.colorbar(im)
        plt.show()

    def rhsfu(self, u, v):
        N, M = np.shape(u)
        n = np.arange(N); n[int(N/2)+1:]-=N
        n = np.array([n for _ in range(M)])
        n = np.transpose(n)
        m = np.arange(M); m[int(M/2)+1:]-=M
        return (self.D_u*(ifft2(-(2*np.pi*n/N)**2*fft2(u)) + ifft2(-(2*np.pi*m/M)**2*fft2(u))) + 
            self.c1 - self.c_*u + self.c3*u**2*v).real

    def rhsfv(self, u, v):
        N, M = np.shape(v)
        n = np.arange(N); n[int(N/2)+1:]-=N
        n = np.array([n for _ in range(M)])
        n = np.transpose(n) 
        m = np.arange(M); m[int(M/2)+1:]-=M
        return (self.D_v*(ifft2(-(2*np.pi*n/N)**2*fft2(v)) + ifft2(-(2*np.pi*m/M)**2*fft2(v))) + 
                self.c2 - self.c3*u**2*v).real

    def rhsf(self, w, t):
        # w is an array with w = [u, v] where u and v are matrices
        w = self.arraytogrids(w)
        u = w[0]
        v = w[1]
        result = np.array([self.rhsfu(u, v), self.rhsfv(u, v)])
        return self.gridstoarray(result)

    def gridstoarray(self,w):
        u = w[0]
        v = w[1]
        wresult = np.zeros(2*self.num_dx*self.num_dy)
        idx = 0
        for i in range(self.num_dx):
            for j in range(self.num_dy):
                wresult[idx] = u[i][j]
                idx += 1

        for i in range(self.num_dx):
            for j in range(self.num_dy):
                wresult[idx] = v[i][j]
                idx += 1
        
        return wresult

    def arraytogrids(self,w):
        idx = 0
        u = np.zeros((self.num_dx,self.num_dy))
        v = np.zeros((self.num_dx,self.num_dy))
        for i in range(self.num_dx):
            for j in range(self.num_dy):
                u[i][j] = w[idx]
                idx += 1

        for i in range(self.num_dx):
            for j in range(self.num_dy):
                v[i][j] = w[idx]
                idx += 1

        return np.array([u,v])

    # returns True if the parameters passed to the function meets the instability 
    # critera for the given reaction function (ex.: "Sch" for Schnakenberg)
    # params = [c_-1, c1, c2, c3, c4, c5, k, D_u, D_v]
    def param_check(self):

        sh1 = (self.c3/self.c_) * (self.c1 + self.c2) # shortcut 1

        # Schnakenberg reaction model
        if self.func == "Sch":

            # criterion 1
            if -self.c_ + (2 * self.c_* self.c2)/(self.c1 + self.c2) - sh1**2 > 0:
                return False
            # criterion 2
            if sh1 < 0:
                return False
            # criterion 3
            if -self.D_u*(sh1**2 / self.c3) + self.D_v*self.c_ + self.D_v*(2 * self.c_ * self.c2)/(self.c1 + self.c2) < 2 * m.sqrt(self.D_u * self.D_v) * m.sqrt(sh1) or 2 * m.sqrt(self.D_u * self.D_v) * m.sqrt(sh1) < 0:
                return False

            return True
        

def main():
    
    grid = Grid()
    grid.integrate(tend=6)


if __name__ == "__main__":
    main()
