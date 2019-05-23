# Class for running pick'n'mix training between 2 agents and a turn based RL environment
# agent1 is drl, agent2 is linear or 'simple'
import random
import gym
import gym_nim
import numpy as np
import matplotlib.pyplot as plt

import Classifier
import SARSAAgent
import DQNEpsilonGreedy
import DoubleDQNEpsilonGreedy
import QLearner
import RandomOptimalAgent

drl_color_map = {
    "DQN": 'blue',
    "DDQN": 'green',
    "SARSA": 'red'
}


class Trainer:
    def __init__(self, environment, agent1, agent2, eval_opponent):
        self.env = environment
        self.a1 = agent1
        self.a2 = agent2
        self.eval_opponent = eval_opponent
        return

    def training_set(self, num_trials, episodes, batch_size=40):
        training_rewards = []
        training_optimalities = []
        for i in range(num_trials):
            r, o = self.training(episodes, i, batch_size)
            training_rewards.append(r)
            training_optimalities.append(o)
        return training_rewards, training_optimalities

    # trainings is a dict map of trainer object to array of result arrays
    @staticmethod
    def generate_statistics(plot_type, trainings, episodes, num_averages, filename):
        # plot object
        fig, ax = plt.subplots()
        plt.xlabel('Episode * 10')
        if plot_type == "reward":
            plt.ylabel('Average Reward')
        elif plot_type == "optimality":
            plt.ylabel('Average Proportion of Optimal Moves')
        opponent = "simple"

        # extract data and build statistics plot
        for k, v in trainings.items():
            zipped = zip(*v)
            maxs = list(map(max, zipped))
            zipped = zip(*v)
            mins = list(map(min, zipped))
            avgs = [0 for _ in range(len(v[0]))]
            xs = [x for x in range(len(v[0]))]

            for t in v:
                for entry in range(len(t)):
                    avgs[entry] += t[entry]
            for i in range(len(v[0])):
                avgs[i] /= num_averages

            color = drl_color_map[k.agent1.type]
            opponent = k.agent2.type

            ax.plot(xs, maxs, color=color, alpha=0.2)
            ax.plot(xs, mins, color=color, alpha=0.2)
            ax.plot(xs, avgs, color=color, alpha=1, label=k.agent1.type)

            plt.legend(loc='best')
            plt.axis([0, episodes / 10, -1, 1])
            ax.fill_between(xs, maxs, mins, where=maxs > mins, facecolor=color, alpha=0.2)

        title = "DRL Agents trained against " + opponent + " agent with " + str(num_averages) + " trials."
        ax.set_title(str(title))
        plt.savefig(filename)
        fig.show()
        return

    def experiment(self, hidden, epsilon1, epsilon2, eps_auto_decay, eps_limit, n, batch_size, selfplay=False):
        fig_title = str(len(hidden)) + " Layers of " + str(hidden[0]) + " Nodes"
        # filename = self.a1.type + "_vs_" + self.a2.type + "_evaluated_" + self.eval_opponent + "_lr" +
        # remove_decimal_point(str(self.a1.learning_rate)) + "_" + str(hidden[0]) + "_" + str(len(hidden)) + ".png"

        wins1 = []
        wins2 = []
        opt_moves1 = np.array([])
        opt_moves2 = np.array([])

        try:
            vs = self.a2.type
        except AttributeError:
            vs = "SELF"
            wins2 = [0.0]
            opt_moves2 = [0.0]
        filename = "../Results/SELFPLAY_" + self.a1.type + "_vs_" + vs + "_evaluated_" + self.eval_opponent.type + \
                   "_lr" + remove_decimal_point(str(self.a1.learning_rate)) + "_" + str(hidden[0]) + "_" + \
                   str(len(hidden)) + "_demo.png"

        print("File: ", filename)

        episode = []



        random.seed(0)
        for j in range(n):
            interval = 25
            if j % interval == 0:
                # ---------- Play Q1 against EVALUATION OPPONENT -----------#
                if eps_auto_decay:
                    # Increase Epsilon over time
                    self.a1.epsilon += interval * (1 - epsilon1) / eps_limit

                x = 250
                self.a1.num_games = 0
                self.a1.won = 0
                self.a1.moves = 0
                self.a1.t = 0
                started = 0

                for i in range(0, x):
                    r = random.randrange(2)
                    if r == 0:
                        started += 1
                        while True:  # Agent first
                            if policy_play1(self.env, self.a1, self.eval_opponent) is False:
                                break
                        self.env.reset()
                    if r == 1:
                        while True:
                            if policy_play2(self.env, self.a1, self.eval_opponent) is False:
                                break
                        self.env.reset()
                episode.append(j)
                wins1.append(self.a1.won / (x - started))
                opt_moves1 = np.append(opt_moves1, self.a1.t / self.a1.moves)

                # if not self play, evaluate the second agent against the eval_opponent
                if selfplay is False:
                    # -------- Play Q2 against SMART -------- #
                    if eps_auto_decay:
                        # increase epsilon for agent 2
                        self.a2.epsilon += interval * (1 - epsilon2) / eps_limit
                    x = 250
                    self.a2.num_games = 0
                    self.a2.won = 0
                    self.a2.moves = 0
                    self.a2.t = 0
                    started = 0
                    for i in range(0, x):
                        r = random.randrange(2)
                        if r == 0:
                            started += 1
                            while True:  # Agent first
                                if policy_play1(self.env, self.a2, self.eval_opponent) is False:
                                    break
                            self.env.reset()
                        if r == 1:
                            while True:
                                if policy_play2(self.env, self.a2, self.eval_opponent) is False:
                                    break
                            self.env.reset()

                    wins2.append(self.a2.won / (x - started))
                    opt_moves2 = np.append(opt_moves2, self.a2.t / self.a2.moves)

            # -------- Q1 vs Q2 TRAINING ---------#
            if j % 100 == 0:
                print("{} training set against {}: Episode: {}/{}, e: {:.2}, Avg Wins1: {:.3}, "
                      "Prev Optimality1: {:.3}, Avg Optimality1: {:.3}, Avg Wins2: {:.3}, Prev Optimality2: {:.3}, "
                      "Avg Optimality2: {:.3}"
                      .format(self.a1.type,
                              vs,
                              j,    # game
                              n,    # total games
                              self.a1.epsilon,  # current eps
                              wins1[-1],        # prev wins 1
                              opt_moves1[-1],   # prev opt 1
                              (np.average(opt_moves1)),     # avg opt 1
                              wins2[-1],        # prev wins 2
                              opt_moves2[-1],   # prev opt 2
                              (np.average(opt_moves2))      # avg opt 2
                              ))
            r = random.randrange(2)
            if r == 0:
                c = 0
                while True:  # self.a1 goes first
                    if selfplay:
                        if two_player1(self.env, self.a1, self.a1, c) is False:
                            break
                    else:
                        if two_player1(self.env, self.a1, self.a2, c) is False:
                            break
                    c += 1
                self.env.reset()

            if r == 1:
                c = 0
                while True:  # self.a2 goes first
                    if selfplay:
                        if two_player2(self.env, self.a1, self.a1, c) is False:
                            break
                    else:
                        if two_player2(self.env, self.a1, self.a2, c) is False:
                            break
                    c += 1
                self.env.reset()

            # perform experience replay if memory exceeds limit
            self.do_replay(batch_size, selfplay)
            # perform target network update based on current episode value j
            self.do_target_update(batch_size, j)

            # if j % 1000 == 0:
            #     print(opt_moves1)
            #     print(opt_moves2)

        print(opt_moves1)
        print(opt_moves2)

        fig = plt.figure()
        fig.suptitle(fig_title)
        plt.plot(episode, opt_moves1, 'r',
                 label='Q1: ' + r'$\alpha=$' + str(self.a1.learning_rate) + r', $\gamma=$' + str(1.0) +
                       r', $\epsilon_i=$' + str(0.8))
        if not selfplay:
            plt.plot(episode, opt_moves2, 'b',
                     label='Q2: ' + r'$\alpha=$' + str(self.a2.learning_rate) + r', $\gamma=$' + str(1.0) +
                           r', $\epsilon_i=$' + str(0.8))
        plt.ylabel("Ratio of winning moves")
        plt.xlabel('Number of training games')
        plt.xticks()
        plt.yticks()
        plt.ylim(0, 1.1)
        plt.xlim(0, n)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.axhline(1, color='grey', linestyle='--')
        plt.ylim(0, 1.1)
        plt.savefig(filename)
        plt.show()

    def do_target_update(self, selfplay, j):
        if self.a1.type == "DDQN":
            if j % 50 == 0:
                self.a1.update_target_model()
        if not selfplay:
            if self.a2.type == "DDQN":
                if j % 50 == 0:
                    self.a2.update_target_model()

    def do_replay(self, batch_size, selfplay):
        if self.a1.type == "DQN" or self.a1.type == "SARSA" or self.a1.type == "DDQN":
            if len(self.a1.memory) > batch_size:
                self.a1.replay(batch_size, self.env)
        if not selfplay:
            if self.a2.type == "DQN" or self.a2.type == "SARSA" or self.a2.type == "DDQN":
                if len(self.a2.memory) > batch_size:
                    self.a2.replay(batch_size, self.env)


def two_player1(env, agent1, agent2, c):
    next_state, reward, done, _, action_picked = agent1.step(env, 0)
    if done:
        agent2.lost(1)
        agent1.won_game(0)
        return False

    if c != 0:
        # Update agent2
        agent2.update(env.state, 1)

    next_state, reward, done, _, action_picked = agent2.step(env, 1)
    if done:
        agent1.lost(0)
        agent2.won_game(1)
        return False

    # Update agent1
    agent1.update(env.state, 0)


def two_player2(env, agent1, agent2, c):
    next_state, reward, done, _, action_picked = agent2.step(env, 1)
    if done:
        agent1.lost(0)
        agent2.won_game(1)
        return False

    if c != 0:
        # Update agent1
        agent1.update(env.state, 0)

    next_state, reward, done, _, action_picked = agent1.step(env, 0)
    if done:
        agent2.lost(1)
        agent1.won_game(0)
        return False

    # Update agent2
    agent2.update(env.state, 1)


def policy_play1(env, agent, opponent):
    """ agent vs Smart"""
    before = env.state[0] ^ env.state[1] ^ env.state[2] ^ env.state[3]
    if before != 0:
        agent.moves += 1

    next_state, reward, done, _, action_picked = agent.step(env, 0)

    after = env.state[0] ^ env.state[1] ^ env.state[2] ^ env.state[3]

    if after == 0:
        agent.t += 1
    if done:
        agent.won += 1
        return False

    s, reward, done, info, _ = opponent.step(env, 0)
    if done:
        return False


def policy_play2(env, agent, opponent):
    """ Smart vs agent """
    s, reward, done, info, _ = opponent.step(env, 0)
    if done:
        return False

    before = env.state[0] ^ env.state[1] ^ env.state[2] ^ env.state[3]
    if before != 0:
        agent.moves += 1

    next_state, reward, done, _, action_picked = agent.step(env, 0)

    after = env.state[0] ^ env.state[1] ^ env.state[2] ^ env.state[3]
    if after == 0:
        agent.t += 1
    if done:
        agent.won += 1
        return False


def remove_decimal_point(string):
    ret_str = ""
    for i in string:
        if i == ".":
            pass
        else:
            ret_str += i
    return ret_str


def main():
    # Environment
    env = gym.make("nim-v0")
    env.set_number_of_heaps(4)
    env.set_heaps_starting_positions([1, 3, 5, 7])
    state_size = env.state_size
    action_size = env.action_size

    # DRL Agent model parameters
    hidden = [100, 100, 100]
    lr = 0.0003
    batch_size = 64

    # q_learn_vs_q_learn(action_size, batch_size, env, hidden, state_size)
    # sarsa_vs_dqn(action_size, batch_size, env, hidden, state_size, lr)
    # ddqn_selfplay(action_size, batch_size, env, hidden, state_size, lr)
    ddqn_vs_classifier(action_size, batch_size, env, hidden, state_size, lr)
    return


def q_learn_vs_q_learn(action_size, batch_size, env, hidden, state_size):
    epsilon1 = 0.8
    a1 = QLearner.QLearner(env.state, state_size, action_size, epsilon1, 0.99, env)

    epsilon2 = 0.8
    a2 = QLearner.QLearner(env.state, state_size, action_size, epsilon2, 0.99, env)

    EVAL_OPPONENT_OPTIMALITY_PROPORTION = 1
    eval_opponent = RandomOptimalAgent.RandomOptimalAgent(EVAL_OPPONENT_OPTIMALITY_PROPORTION, state_size, action_size,
                                                          mem_len=0)
    # number of training games
    number_of_games = 15000
    t = Trainer(env, a1, a2, eval_opponent)
    t.experiment(hidden=hidden, epsilon1=epsilon1, epsilon2=epsilon2, eps_auto_decay=True, eps_limit=10000,
                 n=number_of_games,
                 batch_size=batch_size, selfplay=False)
    return


def sarsa_vs_dqn(action_size, batch_size, env, hidden, state_size, lr):
    epsilon1 = 1.0
    a1 = SARSAAgent.SARSAAgent(state_size, action_size, discount_factor=1.0, learning_rate=lr,
                               exploration_decay=0.9998, hidden_layers=hidden)

    epsilon2 = 1.0
    a2 = DQNEpsilonGreedy.DQNAgent(state_size, action_size, discount_factor=1.0, learning_rate=lr,
                                   exploration_decay=0.999, hidden_layers=hidden)

    EVAL_OPPONENT_OPTIMALITY_PROPORTION = 1
    eval_opponent = RandomOptimalAgent.RandomOptimalAgent(EVAL_OPPONENT_OPTIMALITY_PROPORTION, state_size, action_size,
                                                          mem_len=0)
    # number of training games
    number_of_games = 50000
    t = Trainer(env, a1, a2, eval_opponent)
    t.experiment(hidden=hidden, epsilon1=epsilon1, epsilon2=epsilon2, eps_auto_decay=False, eps_limit=10000,
                 n=number_of_games,
                 batch_size=batch_size, selfplay=False)
    return


def ddqn_selfplay(action_size, batch_size, env, hidden, state_size, lr):
    epsilon1 = 1.0
    a1 = DoubleDQNEpsilonGreedy.DoubleDQNAgent(state_size, action_size, discount_factor=1.0, learning_rate=lr,
                                               exploration_decay=0.999, hidden_layers=hidden)

    EVAL_OPPONENT_OPTIMALITY_PROPORTION = 1
    eval_opponent = RandomOptimalAgent.RandomOptimalAgent(EVAL_OPPONENT_OPTIMALITY_PROPORTION, state_size, action_size,
                                                          mem_len=0)
    # number of training games
    number_of_games = 50000
    t = Trainer(env, a1, None, eval_opponent)
    t.experiment(hidden=hidden, epsilon1=epsilon1, epsilon2=0.0, eps_auto_decay=False, eps_limit=10000,
                 n=number_of_games,
                 batch_size=batch_size, selfplay=True)
    return


def ddqn_vs_classifier(action_size, batch_size, env, hidden, state_size, lr):
    epsilon1 = 1.0
    a1 = DoubleDQNEpsilonGreedy.DoubleDQNAgent(state_size, action_size, discount_factor=1.0, learning_rate=lr,
                                               exploration_decay=0.999, hidden_layers=hidden)
    epsilon2 = 0.0
    a2 = Classifier.MoveClassifier(state_size, action_size, None)
    dataset = eval(open("classifier_partial_sample_data.txt", "r").read())
    # dataset = eval(open("classifier_complete_sample_data.txt", "r").read())
    a2.train(dataset)

    EVAL_OPPONENT_OPTIMALITY_PROPORTION = 1
    eval_opponent = RandomOptimalAgent.RandomOptimalAgent(EVAL_OPPONENT_OPTIMALITY_PROPORTION, state_size, action_size,
                                                          mem_len=0)
    # number of training games
    number_of_games = 50000
    t = Trainer(env, a1, a2, eval_opponent)
    t.experiment(hidden=hidden, epsilon1=epsilon1, epsilon2=epsilon2, eps_auto_decay=False, eps_limit=10000,
                 n=number_of_games,
                 batch_size=batch_size, selfplay=False)
    return


if __name__ == '__main__':
    main()
