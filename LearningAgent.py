import checkers_env
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

class LearningAgent:

    def __init__(self, step_size, epsilon, env):
        '''
        :param step_size:
        :param epsilon:
        :param env:
        '''

        self.step_size = step_size
        self.epsilon = epsilon
        self.env = env
        # self.q_table = np.zeros(len(env.state_space), len(env.action_space))


    def evaluation(self):
        '''
        evaluate the score of the board, i.e., reward function
        '''


    def learning(self):
        '''
        reinforcement learning algorithm
        '''


    def select_action(self):
        '''
        make the movement decision
        '''
        return action








