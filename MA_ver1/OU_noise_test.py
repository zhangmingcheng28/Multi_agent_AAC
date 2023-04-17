# -*- coding: utf-8 -*-
"""
@Time    : 4/17/2023 8:00 PM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.5):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def linear_eps_decay(epsilon_start, epsilon_end, decay_steps, current_step):
    slope = (epsilon_end - epsilon_start) / decay_steps
    intercept = epsilon_start
    current_eps = slope * current_step + intercept
    current_eps = max(current_eps, epsilon_end)
    return current_eps

if __name__ == '__main__':
    # create a figure with two subplots
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(nrows=2, ncols=1)
    minimum_sigma_val = 0.01
    time = 500
    ou = OUNoise(1)  # if action dimension is 1
    ou_states = []
    eps_start = 1
    eps = eps_start
    eps_end = 0.01
    eps_decay_steps = 350
    largest_sigma = 0.2
    for i in range(time):
        #eps = linear_eps_decay(eps_start, eps_end, eps_decay_steps, i)
        sigma = largest_sigma * eps + (1 - eps) * minimum_sigma_val
        eps = max(eps_end, eps - (eps_start - eps_end) / eps_decay_steps)
        print("At time step {}, current eps is {}, the current sigma is {}".format(i, eps, sigma))
        ou.sigma = sigma
        ou_states.append(ou.noise()[0])

    ax[0].plot(ou_states)
    ax[0].set_title('OUNoise with sigma decayed by epsilon-greedy')

    # plot original OU noise for comparison
    ou = OUNoise(1)
    ou_states = []
    for i in range(time):
        ou_states.append(ou.noise()[0])

    ax[1].plot(ou_states)
    ax[1].set_title('Original OUNoise')

    plt.tight_layout()
    plt.show()