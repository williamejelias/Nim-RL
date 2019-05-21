from RLAgent import RLAgent
import numpy as np
import random
import collections


class QLearner(RLAgent):
    def __init__(self, state_setup, state_size, action_size, epsilon, epsilon_decay, env=None):
        super().__init__(state_size, action_size, 0, env)
        self.type = "Q-LEARN"
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = epsilon_decay
        self.action_size = action_size
        self.learning_rate = 0.1
        self.table = self.generate_value_table(state_setup, action_size)

        self.state = None
        self.action = None

    def choose_action_index(self, env):
        state = [str(h) for h in env.state]

        state_string = "".join(state)
        self.state = state_string

        # random move
        if random.uniform(0, 1) > self.epsilon:
            action_index = random.randrange(self.action_size)
            action = env.lookup_action(action_index)
            # Reduce the exploration rate epsilon
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay
        else:
            m = max(self.table[state_string])
            if collections.Counter(self.table[state_string])[m] > 1:  # if more than 1 action w/ max value
                best_action = []
                for i in range(len(self.table[state_string])):
                    if self.table[state_string][i] == m:
                        best_action.append(i)
                action_index = random.choice(best_action)
            else:
                action_index = np.argmax(self.table[state_string])
            action = env.lookup_action(action_index)
        self.action = action_index
        return action, action_index

    def step(self, env, turn):
        chosen_move, chosen_move_index = self.choose_action_index(env)

        # if move is illegal pick again using same policy and store the associated punishment
        while chosen_move_index in env.get_illegal_move_indices():
            # pick again
            self.invalid_move(turn)
            chosen_move, chosen_move_index = self.choose_action_index(env)

        # lookup the bean heap action from index and perform move
        action = env.lookup_action(chosen_move_index)
        next_state, reward, done, info = env.step(action, )

        return next_state, reward, done, info, action

    @staticmethod
    def generate_value_table(state_setup, action_size):
        # e.g. state_setup = [1, 3, 5, 7]
        number_of_states = 1
        for i in state_setup:
            number_of_states *= (i+1)

        all_states = [[str(x)] for x in range(0, state_setup[0]+1)]
        for superset_state in range(1, len(state_setup)):
            new_all_states = []
            s_list = [x for x in range(0, state_setup[superset_state] + 1)]
            for entry in s_list:
                for state_list in all_states:
                    new_all_states.append(state_list + [str(entry)])
            all_states = new_all_states

        value_table = {}
        for state in all_states:
            state_string = "".join(state)
            value_table[state_string] = np.zeros(action_size)
        return value_table

    def won_game(self, turn):
        # print("won")
        reward = 1
        self.table[self.state][self.action] += self.learning_rate * (reward - self.table[self.state][self.action])
        # print(self.state, " and ", self.action, " gets value ", self.table[self.state][self.action])

    def lost(self, turn):
        # print("lost")
        reward = -1
        self.table[self.state][self.action] += self.learning_rate * (reward - self.table[self.state][self.action])
        # print(self.state, " and ", self.action, " gets value ", self.table[self.state][self.action])

    def update(self, next_state, turn):
        next_state_string = "".join([str(x) for x in next_state])
        reward = 0
        s = self.state
        a = self.action
        sp = next_state_string
        self.table[s][a] += self.learning_rate * (reward + np.amax(self.table[sp]) - self.table[s][a])

    def invalid_move(self, turn):
        self.table[self.state][self.action] = -10

    def replay(self, batch, env):
        pass

    def single_fit(self, env, state, action, reward, next_state, done):
        pass
