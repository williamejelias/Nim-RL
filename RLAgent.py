import collections
import gym
import DNN

gym.logger.set_level(40)


class RLAgent:
    def __init__(self, state_size, action_size, mem_len, env):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=mem_len)
        self.env = env

        self.env = None
        self.state0 = None
        self.state1 = None
        self.action_index0 = None
        self.action_index1 = None

        self.moves = 0.0000001
        self.t = 0.0000001
        self.num_games = 0
        self.won = 0

    def build_model(self, lr, hidden, loss='mse'):
        return DNN.DNN(self.state_size, self.action_size, lr, hidden, loss)

    @staticmethod
    def predict(model, state):
        return model.predict(state)

    def step(self, env, turn):
        raise NotImplementedError

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def single_fit(self, env, state, action, reward, next_state, done):
        raise NotImplementedError

    def won_game(self, turn):
        reward = 1
        # save to memory to allow for replay and reduce forgetting
        # next state is None, done is True
        if turn == 0:
            self.remember(self.state0, self.action_index0, reward, None, True)
            self.single_fit(self.env, self.state0, self.action_index0, reward, None, True)
        else:
            self.remember(self.state1, self.action_index1, reward, None, True)
            self.single_fit(self.env, self.state1, self.action_index1, reward, None, True)

    def lost(self, turn):
        reward = -1
        # save to memory to allow for replay and reduce forgetting
        # next state is None, done is True
        if turn == 0:
            self.remember(self.state0, self.action_index0, reward, None, True)
            self.single_fit(self.env, self.state0, self.action_index0, reward, None, True)
        else:
            self.remember(self.state1, self.action_index1, reward, None, True)
            self.single_fit(self.env, self.state1, self.action_index1, reward, None, True)

    def update(self, next_state, turn):
        reward = 0
        # save to memory to allow for replay and reduce forgetting
        if turn == 0:
            self.remember(self.state0, self.action_index0, reward, next_state, False)
            self.single_fit(self.env, self.state0, self.action_index0, reward, next_state, False)
        else:
            self.remember(self.state1, self.action_index1, reward, next_state, False)
            self.single_fit(self.env, self.state1, self.action_index1, reward, next_state, False)

    def invalid_move(self, turn):
        # remember the transition
        if turn == 0:
            action = self.env.get_move_list()[self.action_index0]
            self.remember(self.state0, action, -10, self.state0, False)
            self.single_fit(self.env, self.state0, self.action_index0, -10, self.state0, False)
        else:
            action = self.env.get_move_list()[self.action_index1]
            self.remember(self.state1, action, -10, self.state1, False)
            self.single_fit(self.env, self.state1, self.action_index1, -10, self.state1, False)
