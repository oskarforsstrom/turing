import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

class Simulator:
    """class for simulating"""

    def __init__(self, arr, steps_per_frame = 100):
        self.arr = arr
        # self.grid = grid
        self.steps_per_frame = steps_per_frame


        # Initialize figure for animation
        self.fig = plt.figure()
        self.ax = plt.subplot(xlim=(0, 50), ylim=(0, 50))
        self.im = self.ax.imshow(self.arr, animated=True)
        self.fig.colorbar(self.im)


    def snapshot(self, step) :

        """
            This is an 'auxillary' function needed by animation.FuncAnimation
            in order to show the animation of the 2D Lennard-Jones system
        """
        if step > 0:
            print(self.arr[0,0])
            # self.grid.integrate() 
            self.transform() 
        self.im.set_array(self.arr)
        return self.im,


    def animate(self):

        nn = 1000//self.steps_per_frame 
        anim = animation.FuncAnimation(self.fig, self.snapshot,
            frames=nn, interval=200, blit=True, repeat=False)
        plt.show()  # show the animation

    
    def transform(self):
        self.arr += 0.1*np.ones((50,50))


def main():
    arr = np.zeros((50,50))
    simulator = Simulator(arr=arr)
    simulator.animate()

if __name__ == "__main__":
    main()
