from RLAgent import RLAgent
import collections
import random
import numpy as np


# Deep Q-learning Agent
class DQNAgent(RLAgent):
    def __init__(self, state_size, action_size, discount_factor=1.0, learning_rate=0.01, exploration_decay=0.999,
                 env=None, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [20, 20]
        super().__init__(state_size, action_size, 10000, env)
        self.type = "DQN"
        self.gamma = discount_factor  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exploration_decay
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers

        self.model = self.build_model(self.learning_rate, self.hidden_layers)

    def choose_action(self, model_prediction):
        # print(model_prediction)
        if random.uniform(0, 1) > self.epsilon:
            # The agent randomly selects a move index
            random_move_index = random.randint(0, len(model_prediction) - 1)
            random_move = model_prediction[random_move_index]
            return random_move, random_move_index
        else:
            # extract a_max (greedy move selection)
            m = max(model_prediction)
            if collections.Counter(model_prediction)[m] > 1:  # if more than 1 action w/ max value
                best_action = []
                for i in range(len(model_prediction)):
                    if model_prediction[i] == m:
                        best_action.append(i)
                greedy_move_index = random.choice(best_action)
                greedy_move = model_prediction[greedy_move_index]
            else:
                greedy_move_index = np.argmax(model_prediction)
                greedy_move = model_prediction[greedy_move_index]
            return greedy_move, greedy_move_index

    # same policy but only looks at legal moves
    def choose_legal_action_index(self, env, model_prediction):
        # filter out illegal moves
        poss_values_dict = {}
        poss_values = []
        for pm in env.get_possible_move_indices():
            try:
                poss_values_dict[model_prediction[pm]].append(pm)
            except KeyError:
                poss_values_dict[model_prediction[pm]] = [pm]
            poss_values.append(model_prediction[pm])

        # print("prediction: ", model_prediction)
        # print("poss values: ", poss_values)
        # print("poss values dict: ", poss_values_dict)

        # random select or choose best
        if random.uniform(0, 1) > self.epsilon:
            output = random.choice(env.get_possible_move_indices())
            return output
        else:
            selected = np.amax(poss_values)
            x = random.randint(0, len(poss_values_dict[selected])-1)
            output = poss_values_dict[selected][x]
            return output

    # perform a step in the passed environment
    def step(self, env, turn):
        self.env = env
        state = np.reshape(env.state, [1, env.state_size])
        if turn == 0:
            self.state0 = state
        else:
            self.state1 = state

        # Predict the reward value based on the given state
        act_values = self.predict(self.model, state)

        # pick a move using policy
        chosen_move, chosen_move_index = self.choose_action(act_values)

        already_tried_illegals = []
        # if move is illegal pick again using same policy and store the associated punishment
        count = 0
        while chosen_move_index in env.get_illegal_move_indices():
            count += 1
            # cutoff for illegal tries to save time
            if len(already_tried_illegals) > 3 or count > 3:
                # lookup a legal bean heap action (breaks while loop)
                chosen_move_index = self.choose_legal_action_index(env, act_values)
            else:
                if chosen_move_index in already_tried_illegals:
                    pass
                else:
                    # add to memory with punishment
                    already_tried_illegals.append(chosen_move_index)
                    self.invalid_move(turn)

                # pick again
                chosen_move, chosen_move_index = self.choose_action(act_values)

        # lookup the bean heap action from index and perform move
        if turn == 0:
            self.action_index0 = chosen_move_index
        else:
            self.action_index1 = chosen_move_index
        action = env.lookup_action(chosen_move_index)
        next_state, reward, done, info = env.step(action, )

        # return action result
        return next_state, reward, done, info, action

    def replay(self, batch_size, env):
        # Sample mini-batch from the memory
        mini_batch = random.sample(self.memory, batch_size)

        # Extract information from each memory and fit
        for state, action, reward, next_state, done in mini_batch:
            self.single_fit(env, state, action, reward, next_state, done)

        # Reduce the exploration rate epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def single_fit(self, env, state, action, reward, next_state, done):
        # if done, make our target reward
        target = reward
        if not done:
            # predict the future discounted reward
            next_state = np.reshape(next_state, [1, env.state_size])
            target = reward + self.gamma * np.amax(self.predict(self.model, next_state))

        # make the agent to approximately map the current state to future discounted reward
        # We'll call that target_f
        state = np.reshape(state, [1, env.state_size])
        target_f = self.predict(self.model, state)
        # print(action, env.outputToActionMap)
        # action_index = env.outputToActionMap[action]
        target_f[action] = target

        # Train the Neural Net with the state and target_f
        self.model.model.fit(state, np.array([target_f]), epochs=1, verbose=0)
