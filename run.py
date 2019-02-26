import time
import game
import agent
import random
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
import itertools
import tensorboard
import pickle
import h5py

def train_model(learning_rate=0.01, trace_decay=0.8, discount_factor=1, n_games=1000, n_hidden=40, path='weights.npy'):
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
    # Initialize neural network weights as random normally distributed values, scaled by 2 / sqrt(number of neurons)
    n_inputs = (game.NUMCOLS * 4 + 3) * 2
    w1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2 / (n_inputs))  # Weights input->hidden
    w2 = np.random.rand(n_hidden, 1) * np.sqrt(2 / (n_hidden))          # Weights hidden->output
    b1 = np.zeros(n_hidden, 1)                                          # Bias on hidden layer nodes
    b2 = np.zeros(1,1)                                                  # Bias on output layer node

    # TD-lambda uses eligibility traces to keep track of useful gradients
    trace_w1 = np.zeros_like(w1)
    trace_w2 = np.zeros_like(w2)
    trace_b1 = np.zeros_like(b1)
    trace_b2 = np.zeros_like(b2)

    def sigmoid(z):
        return (1 / (1. + np.exp(-z)))

    # get board state
    x = g.extract_features(player=)

    # Forward pass:
    z1 = np.dot(w1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    V = sigmoid(z2)

    # Backprop
    dV_db2 = V*(1-V)
    dV_dw2 = dV_db2 * a1.T
    dV_db1 = dV_db2 * w2.T * a1 * (1-a1)
    dV_dw1 = dV_db1 * x.T

    # Update eligibility traces
    trace_w1 = trace_decay * discount_factor * trace_w1 + dV_dw1
    trace_w2 = trace_decay * discount_factor * trace_w2 + dV_dw2
    trace_b1 = trace_decay * discount_factor * trace_b1 + dV_db1
    trace_b2 = trace_decay * discount_factor * trace_b2 + dV_db2

    # Step reward
    delta = reward + discount_factor * x_new - x_old

    # Update weights
    w1 += learning_rate * delta * trace_w1
    w2 += learning_rate * delta * trace_w2
    b1 += learning_rate * delta * trace_b1
    b2 += learning_rate * delta * trace_b2





    # Set up new weights etc

    for i_game in range(n_games):
        g = game.new_game()

        players = []
        trace_ih = np.zeros()
        over = False
        playernum = 0
        for t in itertools.count():


            pass

        # Save weights to file every 100th game:
        if i_game % 100 == 0:
            np.savez('weight.npy', w1, w2, b1, b2)

def test(players,numGames=100,draw=False):
    winners = [0,0]
    for _ in range(numGames):
        g = game.Game(game.LAYOUT)
        winner = run_game(players, g, draw)
        print("The winner is : Player %s"%players[not winner].player)
        winners[not winner]+=1
        if draw:
            g.draw()
            time.sleep(10)
    print("Summary:")
    print("Player %s : %d/%d"%(players[0].player,winners[0], sum(winners)))
    print("Player %s : %d/%d"%(players[1].player,winners[1], sum(winners)))


def run_game(players,g,draw=False):
    g.new_game()
    playernum = random.randint(0, 1)
    over = False
    while not over:
        g.extract_features(players[playernum])
        roll = g.roll_dice() # roll dice
        if draw: #draw board
            g.draw(roll)
        playernum = (playernum + 1) % 2  # Switch players
        if playernum: # If black is on turn, reverse board
            g.reverse()
        turn(players[playernum], g, roll, draw)
        if playernum: # If black just took a turn, switch board back to white
            g.reverse()
        over = g.is_done()
        if draw:
            time.sleep(.02)

    return g.winner()


def turn(player, g, roll, draw=False):

    if draw:
        print("Player %s rolled <%d,%d>."%(player.token, roll[0], roll[1]))
    moves = g.getActions(roll, g.players[0], nodups=True)
    if moves:
        move = player.getAction(moves, g)
    else:
        move = None
    if move:
        g.take_action(move, g.players[0])


def load_weights(weights):
    if weights is None:
        try:
            import pickle
            weights = pickle.load(open('weights.bin','r'))
        except IOError:
            print("You need to train the weights to use the better evaluation function")
    return weights


def main(args=None):
    from optparse import OptionParser
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option("-t","--train", dest="train",action="store_true",
                      default=False,help="Train TD Player")
    parser.add_option("-d","--draw",dest="draw",action="store_true",default=False,
                      help="Draw game")
    parser.add_option("-n","--num",dest="numgames",default=1,help="Num games to play")
    parser.add_option("-p","--player1",dest="player1",
                      default="random",help="Choose type of first player")
    parser.add_option("-e","--eval",dest="eval",action="store_true",default=False,
                        help="Play with the better eval function for player")

    (opts,args) = parser.parse_args(args)    

    weights = None

    # if opts.train:
    #     weights = train()
        
    if opts.eval:
        weights = load_weights(weights)

    p1 = None
    if opts.player1 == 'random':
        p1 = agent.RandomAgent(game.Game.TOKENS[0])
    # elif opts.player1 == 'expectimax':
    #     p1 = agent.RLAgent(game.Game.TOKENS[0], evalFn)
    elif opts.player1 == 'human':
        p1 = agent.HumanAgent(game.Game.TOKENS[0])

    p2 = agent.RandomAgent(game.Game.TOKENS[1])

    if p1 is None:
        print("Please specify legitimate player")
        import sys
        sys.exit(1)

    test([p1,p2],numGames=int(opts.numgames),draw=opts.draw)

if __name__=="__main__":
    # main()

    p1 = agent.HumanAgent(game.Game.TOKENS[0])
    # p2 = agent.HumanAgent(game.Game.TOKENS[1])
    # p1 = agent.RandomAgent(game.Game.TOKENS[0])
    p2 = agent.RandomAgent(game.Game.TOKENS[1])
    test([p1, p2], numGames=1, draw=True)
