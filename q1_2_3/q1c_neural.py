#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    ### YOUR CODE HERE: forward propagation
    # z1 = data @ W1 + b1 (1 × H)
    z1 = data @ W1 + b1
    # h = sigmoid(z1) (1 × H)
    h = sigmoid(z1)
    # z2 = h @ W2 + b2 (1 × Dy)
    z2 = h @ W2 + b2
    # y_hat = softmax(z2) (1 × Dy)
    y_hat = softmax(z2)
    # Return probability of correct label
    return y_hat[0, label]
    ### END YOUR CODE


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # z1 = data @ W1 + b1 (M × H) - broadcasting b1
    z1 = data @ W1 + b1
    # h = sigmoid(z1) (M × H)
    h = sigmoid(z1)
    # z2 = h @ W2 + b2 (M × Dy) - broadcasting b2
    z2 = h @ W2 + b2
    # y_hat = softmax(z2) (M × Dy)
    y_hat = softmax(z2)
    # cost = -mean(Σ_i labels[i]·log(y_hat[i]))
    # Softmax ensures y_hat values are in (0,1), so log is safe
    cost = -np.mean(np.sum(labels * np.log(y_hat), axis=1))
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    # From Q1a: gradient w.r.t. softmax input
    # delta_theta = y_hat - labels (M × Dy)
    delta_theta = y_hat - labels
    
    # From Q1b: gradient w.r.t. hidden layer output
    # delta_h = delta_theta @ W2.T (M × H)
    delta_h = delta_theta @ W2.T
    
    # Gradient w.r.t. hidden layer input (before sigmoid)
    # delta_z1 = delta_h ⊙ sigmoid_grad(h) (M × H)
    delta_z1 = delta_h * sigmoid_grad(h)
    
    # Parameter gradients (scaled by 1/M since cost is mean over batch)
    M = data.shape[0]
    # gradW2 = h.T @ delta_theta (H × Dy)
    gradW2 = (h.T @ delta_theta) / M
    
    # gradb2 = sum(delta_theta, axis=0) (1 × Dy) - sum over batch
    gradb2 = np.sum(delta_theta, axis=0, keepdims=True) / M
    
    # gradW1 = data.T @ delta_z1 (Dx × H)
    gradW1 = (data.T @ delta_z1) / M
    
    # gradb1 = sum(delta_z1, axis=0) (1 × H) - sum over batch
    gradb1 = np.sum(delta_z1, axis=0, keepdims=True) / M
    ### END YOUR CODE

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q1c_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR OPTIONAL CODE HERE
    pass
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
