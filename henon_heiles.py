import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


def henon_heiles_potential(x, y):
    """The values of henon heiles poptential (\lambda = 1) for given x/y."""
    return (x**2 + y**2) / 2 + (x**2 * y - y**3 / 3)
    

# generate input data
def generate_input(nsamples=100, x_max=2):
    """Generates random input data with target result values"""
    x = (np.random.rand(nsamples, 1) - 0.5) * x_max
    y = (np.random.rand(nsamples, 1) - 0.5) * x_max

    return np.hstack((x, y)), henon_heiles_potential(x, y).reshape((nsamples, 1))

def preprocess_input(r_raw, z_raw, percent_train=0.7):
    """Reshapes and normalizes input data"""

    index_separate = int(percent_train * len(r_raw))
    
    # split in train/test
    r_train = r_raw[0:index_separate]
    r_test = r_raw[index_separate:]
    z_train = z_raw[0:index_separate]
    z_test = z_raw[index_separate:]

    # normalize
    input_shift = -np.mean(r_train, 0)
    input_factor = 1.0 / np.var(r_train, 0)
    normalize = lambda x: (x + input_shift) * input_factor
    r_train = normalize(r_train)
    r_test = normalize(r_test)

    return (r_train, z_train), (r_test, z_test)



def plot_3d(x, y, z):
    """set up custom surface plot"""   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z)
    plt.show()



def main():
    train, test = preprocess_input(*generate_input())





    



if __name__ == '__main__':
    main()




