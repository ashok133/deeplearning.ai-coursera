
#@authors - Ashok Patel, deeplearning.ai
import sys
import math
import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
# from tensorflow.python.keras.datasets import mnist
from utils import *

np.random.seed(1)

def get_data():
    """
    # Fetch the shapes of input features and output classes; to be used when initializing parameters W and b
    Fetch the data and prepare train_test splits
    """

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()
    print(type(X_train_orig))

    X_train_flatten = X_train_orig.values.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.values.reshape(X_test_orig.shape[0], -1).T

    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    print("X_train_orig" + str(X_train_orig.shape))
    print("X_train_flatten" + str(X_train_flatten.shape))
    print("Y_train" + str(Y_train.shape))
    print("Y_test" + str(Y_test.shape))

    return X_train_flatten, Y_train, X_test_flatten, Y_test

def create_placeholders(n_x, n_y):
    """
    Creating the placeholders for a tensorflow session
    """

    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")

    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow with required shapes
    Layer configuration: [25, 12, 6]
    """

    X_train, Y_train, X_test, Y_test = get_data()

    features_dim = X_train.shape[0]
    labels_dim = Y_train.shape[0]

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, features_dim], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [labels_dim, 12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [labels_dim, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the simple 3-layer model:
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']


    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost (loss)
    """

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

def model(X_train, Y_train, X_test, Y_test, num_epochs, learning_rate, minibatch_size, print_cost = True):
    """
    Implementing a three-layer neural network to classify hand signs (alphabets): LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
    Optimizer - Adaptive momentum (Adam)
    Regularization - None
    Batch size - user input
    Iterations - user input
    LR - user input

    Steps:
    1. Init Parameters
    2. Repeat for each iteration:
        Repeat for each mini_batch:
            1. Foward prop
            2. Compute loss
            3. Backward pass
            4. Update params
    3. Return params
    """
    # saver = tf.train.Saver()                          # to save the learned model

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.0
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Weights saved in 'parameters'!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

        # saver.save(sess, 'learned_model/learned_model')


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_data()
    iters = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    minibatch_size = int(sys.argv[3])
    # get_data()
    trained_params = model(X_train, Y_train, X_test, Y_test, iters, learning_rate, minibatch_size)
    with open('learned_model/params.pickle', 'wb') as handle:
        pickle.dump(trained_params, handle)
