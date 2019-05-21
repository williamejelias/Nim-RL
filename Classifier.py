import collections
import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from RLAgent import RLAgent

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


class MoveClassifier(RLAgent):
    def __init__(self, state_size, action_size, classes, env=None):
        super().__init__(state_size, action_size, 10000, env)
        # data is an array -> [[data, label]]
        self.type = "CLASSIFIER"
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.learning_rate = 0
        self.classes = classes

    def build_model(self, **kwargs):
        # input to model is current game state
        # output of model is probability of each action being optimal
        model = Sequential()
        model.add(Dense(100, input_dim=self.state_size, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(100, activation='relu', kernel_initializer='random_normal'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='random_normal'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, dataset):
        X = np.array([d[0] for d in dataset])
        Y = np.array([d[1] for d in dataset])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print("[INFO] training network...")
        self.model.fit(X_train, Y_train, batch_size=10, epochs=10)

    @staticmethod
    def choose_action(model_prediction):
        m = max(model_prediction)
        if m > 0.5:
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

        else:
            random_move_index = random.randint(0, len(model_prediction) - 1)
            random_move = model_prediction[random_move_index]
            return random_move, random_move_index

    # same policy but only looks at legal moves
    @staticmethod
    def choose_legal_action_index(env, model_prediction):
        # filter out illegal moves
        poss_values_dict = {}
        poss_values = []
        for pm in env.get_possible_move_indices():
            try:
                poss_values_dict[model_prediction[pm]].append(pm)
            except KeyError:
                poss_values_dict[model_prediction[pm]] = [pm]
            poss_values.append(model_prediction[pm])

        # randomly select a valid move
        output = random.choice(env.get_possible_move_indices())
        return output

    def step(self, env, turn):
        state = np.reshape(env.state, [1, env.state_size])

        act_values = self.model.predict(state)[0]

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
        action = env.lookup_action(chosen_move_index)
        next_state, reward, done, info = env.step(action, )

        # return action result
        return next_state, reward, done, info, action

    def single_fit(self, env, state, action, reward, next_state, done):
        pass

    def won_game(self, turn):
        pass

    def lost(self, turn):
        pass

    def update(self, next_state, turn):
        pass

    def invalid_move(self, turn):
        pass

    def replay(self, batch_size, env):
        pass

# mc = MoveClassifier(4, 16, None)
# mc.train(dataset)
