import time
from game import Game
from agent import RandomAgent, HumanAgent, RLAgent
import json
import sys
import multiprocessing as mp
import numpy as np
import itertools

num_eps = 10000
test_eps = set(list(range(1000, num_eps+1, 1000)) + [1, 100, 250, 500, 750, 1500, 2500, 3500, 4500])

def train_model(learning_rate: float=0.01,
                trace_decay: float=0.9,
                num_episodes: int=num_eps,
                n_hidden: int=40,
                method='modified',
                weights=None,
                test_episodes:list=test_eps,
                test_games: int = 400,
                save=False,
                name='file') -> dict:

    """
    Train the neural network weights by means of TD(lambda).
    :param learning_rate: Parameter to govern the size of update steps, float > 0 (alpha in literature)
    :param trace_decay: Parameter to govern how much of the episode history is used to assign credit to TD updates
    :param num_episodes: How many episodes will be trained for
    :param n_hidden:  Amount of neurons in the hidden layer of the network
    :param method: Method to be used in extracting binary features from the current game state. Options: 'original', 'modified'.
    :param weights: Gives a set of already-used weights to start training from a given setpoint or with pre-defined starting weights
    :param test_episodes: List of integers. At these episodes, training will be paused to test the strength of current weights by playing
        a set number of games (test_games parameter) against a default opponent
    :param test_games: How many games will each test consist of?
    :param save: Save weights to file at regular intervals? False by default
    :param name: identifier used to define the save paths of both the weights and the results. Each will automatically
        be appended by the correct extension.

    :return: dictionary of win percentages against the predefined opponent at eacb of the test intervals
    """

    if weights is not None:
        w1, w2, b1, b2 = weights
    else:
        # Initialize neural network weights as random normally distributed values, scaled by 2 / sqrt(number of neurons)
        n_inputs = (Game.NUMCOLS * 4 + 3) * 2
        w1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / (n_inputs))  # Weights input->hidden
        w2 = np.random.randn(n_hidden, 1) * np.sqrt(2 / (n_hidden))         # Weights hidden->output
        b1 = np.zeros((1, n_hidden))                                        # Bias on hidden layer nodes
        b2 = np.zeros((1,1))                                                # Bias on output layer node
        weights = [w1, w2, b1, b2]

    #  NN activation function:
    def sigmoid(z):
        return (1 / (1. + np.exp(-z)))

    #stats = {'episode_lengths': np.zeros(num_episodes), 'episode_winners': np.zeros(num_episodes)}
    test_results = {}
    players = [RLAgent(Game.TOKENS[0], method=method, weights=weights),
               RLAgent(Game.TOKENS[1], method=method, weights=weights)]
    if test_episodes is None:
        test_episodes = []


    for i_episode in range(1, num_episodes+1):

        if (i_episode % 100) == 0:
            # Output where we are now and save the weights to file
            if save:
                np.savez('weights/w_'+name+'.npz', w1=w1, w2=w2, b1=b1, b2=b2)
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()


        if i_episode in test_episodes:
            # Test the current strength of the RLAgent's weights by playing a certain amount of games vs random opponent
            testplayers = [RLAgent(token=Game.TOKENS[0],  method=method, weights=weights),
                           RandomAgent(token=Game.TOKENS[1])]
            current_winrate = test(testplayers, num_games=test_games)
            test_results[i_episode] = current_winrate
            if save:
                with open('results_' + name + '.json', 'w') as f:
                    json.dump(test_results, f)

        # Reset eligibility traces to zero
        trace_w1 = np.zeros_like(w1)
        trace_w2 = np.zeros_like(w2)
        trace_b1 = np.zeros_like(b1)
        trace_b2 = np.zeros_like(b2)

        # Start a new game, get state and evaluation:
        g = Game.new_game()
        x = g.extract_features(player=players[g.playernum].token, method=method)
        V = players[0].evaluate_state(x)

        #  Play out game, observe state transitions and rewards, update weights and traces
        for t in itertools.count():

            reward, done = g.step(players)
            x_new = g.extract_features(player=players[g.playernum].token, method=method)

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

        #stats['episode_lengths'][i_episode] = t
        #stats['episode_winners'][i_episode] = reward

    return test_results


def test(players, num_games=100, graphics=False, log=False):
    """
    Compare the performance of two players over a set number of games.
    :param players: list of two instances of Agent child-classes, will play against each other
    :param num_games: amount of games that will be played
    :param draw: if True, display games on screen
    :param log: display summary of test results
    :return: percentage of games that Player 0 (white) has won
    """
    winners = [0,0]
    if log:
        print("Starting test run of %d games... "%num_games)
    for i_game in range(num_games):
        g = Game.new_game(graphics=graphics)
        winner = g.play(players)
        winners[winner]+=1
        if graphics:
            time.sleep(10)
    winrate_white = winners[0] / sum(winners)
    if log:
        print("Summary:")
        print("Player %s : %d/%d"%(players[0].token, winners[0], sum(winners)))
        print("Player %s : %d/%d"%(players[1].token, winners[1], sum(winners)))

    return winrate_white


def RL_player_from_saved_weights(token, identifier):
    """
    Extract weights from saved .npz file to create a trained RLAgent
    :param token: Will this player play as white or black? (Game.TOKENS[0] or [1])
    :param identifier: identifies saved weight. agent weights will be saved as 'w_identifier.npz'
    :return: trained RLAgent player
    """
    data = np.load('weights/'+identifier+'.npz')
    weights = [data['w1'], data['w2'], data['b1'], data['b2']]
    player = RLAgent(token=token, weights=weights)

    return player


def battle_trained_players(fightlist, filepath):

    """
    Tests a number of trained RL agents against each other by having them fight 1000 matches. Saves result to .json
    :param fightlist: list of tuples,
    """
    results = {}

    for f in fightlist:
        p1 = RL_player_from_saved_weights('o', f[0])
        p2 = RL_player_from_saved_weights('x', f[1])
        players = [p1, p2]
        print("%s  vs  %s" % (f[0], f[1]))
        result = test(players, num_games=1000, log=True)
        results[f[0]+'-'+f[1]] = result

    with open(filepath, 'w') as f:
        json.dump(results, f)

def play_trained_agent(identifier):
    """
    Play a game (as human agent) vs a trained RL agent
    :param identifier: identifies saved weight. agent weights will be saved as 'w_identifier.npz'
    :return:
    """
    players = [HumanAgent(Game.TOKENS[0]), RL_player_from_saved_weights('x', identifier)]
    test(players, num_games=1, graphics=True, log=True)

def train_test_lambdas(starting_weights):

    tdlist = [0.975, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    pathss = ['t0975', 't0950', 't0900', 't0850', 't0800', 't0700',
              't0600', 't0500', 't0400', 't0300', 't0200', 't0100']
    learning_rate = 0.01
    num_episodes = 10000

    n_hidden = 40

    # Use 4 cores at a time for three sets to make training manageable overnight
    for l in range(0, 3):
        jobs = []
        for i in range(0 + 4 * l, 4 + 4 * l):
            process = mp.Process(target=train_model,
                                 args=(learning_rate, tdlist[i], num_episodes, n_hidden, 'modified', starting_weights,
                                       True, 400, test_eps, True, pathss[i]))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print("Set %d/3 done " % (l + 1))

def train_test_n_hidden(starting_weights):

    n_hidden_list = [10, 20, 30, 40, 50, 60, 70, 80]
    paths = ['h10', 'h20', 'h30', 'h40', 'h50', 'h60', 'h70', 'h80']

    learning_rate = 0.01
    num_episodes = 20000
    lamda = 0.8
    test_episodes = set(list(range(1000, num_episodes+1, 1000)) + [1, 100, 250, 500, 750, 1500, 2500, 3500, 4500])
    # Use 4 cores at a time for three sets to make training manageable overnight
    for l in range(0, len(n_hidden_list)//4):
        jobs = []
        for i in range(0 + 4 * l, min(4 + 4 * l, 7)):
            process = mp.Process(target=train_model,
                                 args=(learning_rate, lamda, num_episodes, n_hidden_list[i], 'modified', starting_weights,
                                       True, 400, test_episodes, True, paths[i]))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print("Set %d/3 done " % (l + 1))

def train_test_alphas_lamdas(starting_weights):


    num_games = 4000
    learning_rate_list = [0.015, 0.02, 0.03, 0.04]
    lamda_list = [0.975, 0.9, 0.8, 0.6, 0.4]
    argslist = list(itertools.product(learning_rate_list, lamda_list))
    paths = ['al_' + str(x) + '-' + str(y) for x, y in (argslist)]

    num_sets = len(argslist)//4
    # Use 4 cores at a time for three sets to make training manageable overnight
    for l in range(0, num_sets):
        jobs = []
        for i in range(0 + 4*l, 4 + 4*l):
            process = mp.Process(target=train_model,
                                 args=(argslist[i][0], argslist[i][1], num_games, 40, 'modified', starting_weights,
                                       True, 1000, test_eps, [num_games], True, paths[i]))
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        print("Set %d/%d done " % (l + 1, num_sets))

def train_compare_methods(starting_weights):

    tdlist = [0.975, 0.95, 0.9, 0.85]
    pathss = ['t0975', 't0950', 't0900', 't0850']
    learning_rate = 0.01
    num_episodes = 10000
    test_eps = set(list(range(1000, num_episodes + 1, 1000)) + [1, 100, 250, 500, 750, 1500, 2500, 3500, 4500])

    n_hidden = 40
    # Use 4 cores at a time for three sets to make training manageable overnight
    t1 = time.time()
    jobs = []
    for i in range(0, 4):
        process = mp.Process(target=train_model,
                             args=(learning_rate, tdlist[i], num_episodes, n_hidden, 'modified', starting_weights,
                             True, 400, test_eps, True, pathss[i]))
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print("Set 1 done in %d s"%(time.time()-t1))

    t2 = time.time()
    paths = ['ot0975', 'ot0950', 'ot0900', 'ot0850']
    jobs = []
    for i in range(0, 4):
        process = mp.Process(target=train_model,
                             args=(learning_rate, tdlist[i], num_episodes, n_hidden, 'original', starting_weights,
                             True, 400, test_eps, True, paths[i]))
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    print("Set 2 done in %d s"%(time.time()-t2))


if __name__ == "__main__":

    np.random.seed(0)
    #data = np.load('starting_w.npz')
    #starting_weights = [data['w1'], data['w2'], data['b1'], data['b2']]
    #train_compare_methods(starting_weights)

    play_trained_agent('run 2/w_t0975')