import random
import game
import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
import itertools
import tensorboard
import pickle
import h5py

try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("No module pygame, use command line to play")

import copy

OFF = game.Game.OFF
ON = game.Game.ON




class Agent:
    """ Base class for different players """
    def __init__(self, token):
        self.token = token
    
    def getAction(self,moves,game=None):
        raise NotImplementedError("Override me")

class RandomAgent(Agent):
    def getAction(self, moves, game=None):
        if moves:
            return random.choice(list(moves))
        return None

class HumanAgent(Agent):

    def getAction(self,moves,game=None):
        loc = None
        movesLeft = copy.deepcopy(moves)
        pmove = []
        while True:
            # if no more moves we break
            if not movesLeft:
                break

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if loc is not None:
                        # check to see if we can move the piece
                        newLoc = game.gridLocFromPos(pos,self.token)
                        if newLoc is not None:
                            move = (loc,newLoc)
                            moveLegit = False
                            newMoves = set()
                            for m in list(movesLeft):
                                if m[0]==move:
                                    moveLegit = True
                                    if m[1:]:
                                        newMoves.add(m[1:])
                            # if the move is legit we move it
                            if moveLegit:
                                pmove.append(move)
                                game.take_action((move,), self.token)
                                game.draw()
                                movesLeft = newMoves
                                loc = None
                            else:
                                loc = newLoc
                    else:
                        # get a location to move
                        loc = game.gridLocFromPos(pos,self.token) # TODO implement this

    def getActionCommandLine(self,moves,game=None):
        while True:
            if not moves:
                input("No moves for you...(hit enter)")
                break
            while True:
                mv1 = input("Please enter a move <location start,location end> ('%s' for off the board): "%OFF)
                mv1 = self.get_formatted_move(mv1)
                if not mv1:
                    print('Bad format enter e.g. "3,4"')
                else:
                    break

            while True:
                mv2 = input("Please enter a second move (enter to skip): ")
                if mv2 == '':
                    mv2 = None
                    break
                mv2 = self.get_formatted_move(mv2)
                if not mv2:
                    print('Bad format enter e.g. "3,4"')
                else:
                    break

            if mv2:
                move = (mv1,mv2)
            else:
                move = (mv1,)

            if move in moves:
                break
            elif move[::-1] in moves:
                move = move[::-1]
                break
            else:
                print("You can't play that move")
        return move

    def get_formatted_move(self,move):
        try:
            start,end = move.split(",")
            if start != ON:
                start = int(start)
            if end != OFF:
                end = int(end)
            return (start,end)
        except:
            return False

# class RLAgent(Agent):
#
#     def __init__(self, token, weights=None):
#         super().__init__(token)
#         self.weights = weights
#
#     def load_weights(self, path='weights.npy'):
#         pass
#
#     def evaluate_state(self, state):
#
#
#     def get_action(self, actions, game):
#
#         """
#         Return best action according to self.evaluationFunction,
#         with no lookahead.
#         """
#         bestV = 0
#
#         for a in actions:
#             ateList = game.take_action(a, self.token)
#             features = game.extract_features((game,game.opponent(self.token)))
#             hiddenAct = 1/(1+np.exp(-(self.w1.dot(features)+self.b1)))
#             v = 1/(1+np.exp(-(self.w2.dot(hiddenAct)+self.b2)))
#             if v>bestV:
#                 action = a
#                 bestV = v
#             game.undo_action(a, self.token, ateList)
#
#         return action