import time
from game import Game
from agent import RandomAgent, HumanAgent, RLAgent
import pickle
import json
import sys
import multiprocessing as mp
import random
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
import itertools
import tensorboard
import pickle
import h5py

def train_model(learning_rate=0.01, trace_decay=0.9, num_episodes=10000, n_hidden=40, weights=None, name='file'):
    """
    Trains a simple 3 layer neural network to play backgammon by means of reinforcement learning, using the TD(lambda) algorithm
    :param learning_rate:
    :param trace_decay:
    :param num_episodes:
    :param n_hidden:
    :param weights:
    :param name:
    :return:
    """

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
    test_results = {}
    players = [RLAgent(Game.TOKENS[0], weights=weights), RLAgent(Game.TOKENS[1], weights=weights)]

    test_episodes = [0, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]

    for i_episode in range(1, num_episodes+1):

        if (i_episode % 100) == 0:
            # Output where we are now and save the weights to file
            np.savez('weights_'+name+'.npz', w1=w1, w2=w2, b1=b1, b2=b2)
            print("\rEpisode {}/{}".format(i_episode+1, num_episodes), end="")
            sys.stdout.flush()

        if i_episode in test_episodes:
            testplayers = [RLAgent(token=Game.TOKENS[0], weights=weights), RandomAgent(token=Game.TOKENS[1])]
            current_winrate = test(testplayers)
            test_results[i_episode] = current_winrate
            with open('results_' + name + '.json', 'w') as f:
                json.dump(test_results, f)

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
            delta = (reward - V) if done else (V_new - V)
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


if __name__ == "__main__":

    run_multi = 1
    np.random.seed(0)

    if run_multi:
        tdlist = [0.975, 0.95, 0.9, 0.85,  0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]
        pathlist =['trace0975', 'trace095','trace09','trace085','trace08','trace07',
                   'trace06','trace05','trace04','trace03','trace02','trace01','trace00']

        jobs = []
        for i in range(0, 4):
            process = mp.Process(target=train_model, args=(0.01, tdlist[i], 10000, 40, None, pathlist[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print("Set 1/3 done ")

        jobs = []
        for i in range(4, 8):
            process = mp.Process(target=train_model, args=(0.01, tdlist[i], 10000, 40, None, pathlist[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print("Set 2/3 done")

        jobs = []
        for i in range(8, 12):
            process = mp.Process(target=train_model, args=(0.01, tdlist[i], 10000, 40, None, pathlist[i]))
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        print("Jobs done!!!!!")

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
