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

def nn_layer(input_tensor, input_dim, output_dim, activation_fun=tf.nn.relu):
    """A layer of neurons in a neural network."""
    weights = tf.Variable(tf.truncated_normal([input_dim, output_dim]))
    biases = tf.Variable(tf.zeros([output_dim]) + 0.1)
    
    preactive = tf.matmul(input_tensor, weights) + biases

    if activation_fun is None:
        activation = preactive
    else:
        activation = activation_fun(preactive)

    return activation, weights, biases




def main():

    # gather training and validation data
    train, test = preprocess_input(*generate_input())

    sess = tf.Session()

    #--- setup the network ---
    r_ = tf.placeholder(dtype=tf.float32)
    z_ = tf.placeholder(dtype=tf.float32)
    num_layer1_nodes = 4

    # the actual network
    layer1, _, _ = nn_layer(r_, 2, num_layer1_nodes)
    output, _, _ = nn_layer(layer1, num_layer1_nodes, 1)
    #---

    # setup training
    cost = tf.losses.mean_squared_error(z_, output)
    optimizer = tf.train.AdamOptimizer()  
    training = optimizer.minimize(cost)

    #--- do the training and visualize training and validation costs ---
    sess.run(tf.global_variables_initializer())

    # for plotting
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscaley_on(True)
    line1, = ax.semilogy([], [], label="train")
    line2, = ax.semilogy([], [], label="test")
    plt.xlabel("steps / 1"); plt.ylabel("error / 1"); plt.legend()
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    training_cost = []
    validation_cost = []

    max_iterations = 1000
    delta_convergence = 1e-4

    # do actual training and plotting
    for i in range(max_iterations):
        # train and calulate costs
        sess.run(training, {r_: train[0], z_: train[1]})
        training_cost.append(sess.run(cost, {r_: train[0], z_: train[1]}))
        validation_cost.append(sess.run(cost, {r_: test[0], z_: test[1]}))

        print(training_cost[-1], validation_cost[-1])

        # visualize 
        line1.set_data(np.arange(0, len(training_cost)), training_cost)
        line2.set_data(np.arange(0, len(training_cost)), validation_cost)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > 1:
            if np.abs(training_cost[-1] - training_cost[-2]) < delta_convergence:
                print("---------------------------------\n")
                print("Iteration stopped after {0} steps.\n".format(i))
                break
    #---

    print("\n\nFinal training error: {0}\nFinal validation error{1}".format(
        training_cost[-1],
        validation_cost[-1]
    ))




if __name__ == '__main__':
    main()




