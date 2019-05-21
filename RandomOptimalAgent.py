from RLAgent import RLAgent
import functools
import random
import numpy as np


class RandomOptimalAgent(RLAgent):
    def __init__(self, p, state_size, action_size, mem_len, env=None):
        super().__init__(state_size, action_size, mem_len, env)
        self.p = p
        if p == 1:
            self.type = "OPTIMAL"
        elif p == 0:
            self.type = "RANDOM"
        else:
            self.type = "VARIABLY_OPTIMAL"

    def step(self, env, turn):
        r = random.random()
        if r > self.p:
            return self.random_step(env)
        else:
            return self.optimal_step(env)

    @staticmethod
    def optimal_step(env):
        # first choose heap with non-zero size
        state = env.state.copy()
        # print(state)
        nim_sum = functools.reduce(lambda x, y: x ^ y, state)

        # Calc which move to make
        heap_number, beans_number = None, None
        for index, heap in enumerate(state):
            target_size = heap ^ nim_sum
            if target_size < heap:
                heap_number = index
                beans_number = heap - target_size

        if heap_number is None or beans_number is None:
            return RandomOptimalAgent.random_step(env)

        next_state, reward, done, info = env.step([heap_number, beans_number], )
        return next_state, reward, done, info, None

    @staticmethod
    def random_step(env):
        # first choose heap with non-zero size
        non_zero_heaps = np.arange(4)[env.state != 0]

        #  zero index heap number
        heap_number = np.random.choice(non_zero_heaps)

        # size of the chosen heap
        heap_size = env.state[heap_number]

        # take a random number of beans, of course fewer than the heap size
        beans_number = np.random.choice(heap_size) + 1  # np.random.choice(heapsize) starts at zero, therefore + 1
        next_state, reward, done, info = env.step([heap_number, beans_number], )

        return next_state, reward, done, info, None

    # perform optimal step in passed environment, assuming this environment is nim
    def single_fit(self, env, state, action, reward, next_state, done):
        pass

    def won_game(self, turn):
        return

    def lost(self, turn):
        return

    def update(self, state, turn):
        return

    def invalid_move(self, turn):
        pass

