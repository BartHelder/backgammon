import time
from game import Game
from agent import RandomAgent, HumanAgent, RLAgent

import sys
from multiprocessing import Process
import random
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
import itertools
import tensorboard
import pickle
import h5py

def train_model(learning_rate=0.01, trace_decay=0.9, num_episodes=10000, n_hidden=40, weights=None, path='weights.npz'):

    """
    Trains a simple 3 layer neural network to play backgammon by means of reinforcement learning, using the TD(lambda) algorithm
    :param learning_rate:
    :param trace_decay:
    :param discount_factor:
    :param n_games:
    :param n_hidden: number of neurons in the hidden layer
    :param path:
    :return:
    """

    ## Neural network woo
    if weights is not None:
        w1, w2, b1, b2 = weights
    else:
        # Initialize neural network weights as random normally distributed values, scaled by 2 / sqrt(number of neurons)
        n_inputs = (Game.NUMCOLS * 4 + 3) * 2
        w1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / (n_inputs))  # Weights input->hidden
        w2 = np.random.randn(n_hidden, 1) * np.sqrt(2 / (n_hidden))          # Weights hidden->output
        b1 = np.zeros((1, n_hidden))                                        # Bias on hidden layer nodes
        b2 = np.zeros((1,1))                                                # Bias on output layer node
        weights = [w1, w2, b1, b2]

    def sigmoid(z):
        return (1 / (1. + np.exp(-z)))

    stats = {'episode_lengths': np.zeros(num_episodes), 'episode_winners': np.zeros(num_episodes)}
    test_results = []
    players = [RLAgent(Game.TOKENS[0], weights=weights), RLAgent(Game.TOKENS[1], weights=weights)]

    for i_episode in range(num_episodes):

        if (i_episode + 1) % 100 == 0:
            # Output where we are now and save the weights to file
            np.savez(path, w1=w1, w2=w2, b1=b1, b2=b2)
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()

        if (i_episode) % 1000 == 0:
            testplayers = [RLAgent(token=Game.TOKENS[0], weights=weights), RandomAgent(token=Game.TOKENS[1])]
            current_winrate = test(testplayers)
            test_results.append(current_winrate)

        # Reset eligibility traces to zero
        trace_w1 = np.zeros_like(w1)
        trace_w2 = np.zeros_like(w2)
        trace_b1 = np.zeros_like(b1)
        trace_b2 = np.zeros_like(b2)

        # Start a new game, get state and evaluation:
        g = Game.new_game()
        x = g.extract_features(player=players[g.playernum].token, method='modified')
        V = players[0].evaluate_state(x)

        for t in itertools.count():

            reward, done = g.step(players)
            x_new = g.extract_features(player=players[g.playernum].token, method='modified')

            # Forward pass
            a1 = sigmoid(np.dot(x_new, w1) + b1)  # activation of hidden layer neurons
            V_new = sigmoid(np.dot(a1, w2) + b2)  # activation of output neuron

            # Backprop to get the gradients
            dV_db2 = V_new * (1 - V_new)   # scalar
            dV_dw2 = dV_db2 * a1.T  # scalar * (1x40 vector)' = 40x1 vector
            dV_db1 = dV_db2 * w2.T * (a1 * (1 - a1))   # scalar  * (1x40 vector) = 1x40 vector
            dV_dw1 = x.T * dV_db1  # (1x198 vector transposed * 1x40 vector = 198x40 weights matrix

            # Update eligibility traces
            trace_w1 = trace_decay * trace_w1 + dV_dw1
            trace_w2 = trace_decay * trace_w2 + dV_dw2
            trace_b1 = trace_decay * trace_b1 + dV_db1
            trace_b2 = trace_decay * trace_b2 + dV_db2

            # Step reward
            delta = V_new - V if not done else reward - V
            update = learning_rate * delta

            # Update weights
            w1 += update * trace_w1
            w2 += update * trace_w2
            b1 += update * trace_b1
            b2 += update * trace_b2

            # Next step:
            if done:
                break
            V = V_new
            x = x_new

        stats['episode_lengths'][i_episode] = t
        stats['episode_winners'][i_episode] = reward

    return stats, test_results


def test(players, numGames=100, draw=False):

    winners = [0,0]
    print("Starting test run... ")
    for _ in range(numGames):
        g = Game.new_game(graphics=draw)
        winner = g.play(players)
        #print("The winner is : Player %s"%testplayers[winner].token)
        winners[winner]+=1
        if draw:
            time.sleep(10)
    winrate_white = winners[0] / sum(winners)
    print("Summary:")
    print("Player %s : %d/%d"%(players[0].token, winners[0], sum(winners)))
    print("Player %s : %d/%d"%(players[1].token, winners[1], sum(winners)))

    return winrate_white


if __name__=="__main__":
    # main()

    # p1 = agent.HumanAgent(game.Game.TOKENS[0])
    # # p2 = agent.HumanAgent(game.Game.TOKENS[1])
    # # p1 = agent.RandomAgent(game.Game.TOKENS[0])
    # p2 = agent.RandomAgent(game.Game.TOKENS[1])
    # test([p1, p2], numGames=1, draw=True)

    run_multi = 0
    np.random.seed(0)

    if run_multi:
        tdlist = [0.95, 0.9, 0.8, 0.7]
        pathlist =['weights_trace095.npz','weights_trace09.npz','weights_trace08.npz','weights_trace07.npz']

        jobs = []
        for i in range(0, 4):
            out_list = list()
            process = Process(target=train_model, args=(0.01, tdlist[i], 10, 40, None, pathlist[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print("done!")

    else:

        stats, tr = train_model()
        # data = np.load('weights_trace08.npz')
        # w1 = data['w1']
        # w2 = data['w2']
        # b1 = data['b1']
        # b2 = data['b2']
        # weights = [w1, w2, b1, b2]
        #
        # players = [RLAgent(Game.TOKENS[0], weights=weights), RandomAgent(Game.TOKENS[1])]
        # test(players)
