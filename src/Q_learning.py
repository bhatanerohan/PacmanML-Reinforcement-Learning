import numpy as np
import random
from Agent import *


# Q-learning agent
class QLearningAgent(Agent):
    def __init__(self, color, is_hostile, position, sprite, learning_rate=0.01, discount_factor=0.9, exploration_prob=0.6):
        super().__init__(color, is_hostile, position, sprite)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.n_actions = []
        self.q_values = {}
        self._sprite = sprite

    def get_sprite(self):
        return self._sprite

    def set_n_actions(self, actions):
        self.n_actions = actions

    # def init_q_values(self, maze_size, all_actions):
    #     for row in range(maze_size):
    #         for col in range(maze_size):
    #             for action in all_actions:
    #                 self.q_values[((row, col), action)] = 0

    def get_q_value(self, state, action):
        if (state, action) not in self.q_values.keys():
            return 0
        return self.q_values.get((state, action))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(self.n_actions)
        else:
            q_values = [self.get_q_value(state, action) for action in self.n_actions]
            return self.n_actions[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max(self.n_actions, key=lambda a: self.get_q_value(next_state, a))
        new_q_value = (1 - self.learning_rate) * self.get_q_value(state, action) + \
                      self.learning_rate * (
                              reward + self.discount_factor * self.get_q_value(next_state, best_next_action))
        self.q_values[(state, action)] = new_q_value
        self.exploration_prob = max(self.exploration_prob * self.epsilon_decay, self.epsilon_min)
