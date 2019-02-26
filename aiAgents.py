import agent
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorboard






class ExpectiMiniMaxAgent(agent.Agent):

    def miniMaxNode(self,game,player,roll,depth):
        actions = game.getActions(roll,player,nodups=True)
        rollScores = []

        if player==self.player:
            scoreFn = max
        else:
            scoreFn = min
            depth -= 1

        if not actions:
            return self.expectiNode(game,game.opponent(player),depth)
        for a in actions:
            ateList = game.take_action(a, player)
            rollScores.append(self.expectiNode(game,game.opponent(player),depth))
            game.undo_action(a, player, ateList)

        return scoreFn(rollScores)

    def expectiNode(self,game,player,depth):
        if depth==0:
            return self.evaluationFunction((game,player),self.evaluationArgs)

        total = 0
        for i in range(1,game.die+1):
            for j in range(i+1,game.die+1):
                score = self.miniMaxNode(game,player,(i,j),depth)
                if i==j:
                    total += score
                else:
                    total += 2*score
            
        return total/float(game.die**2)

    def getAction(self, actions, game):
        depth = 1
        if len(actions)>100:
            depth = 0
        outcomes = []
        for a in actions:
            ateList = game.take_action(a, self.player)
            score = self.expectiNode(game,game.opponent(self.player),depth)
            game.undo_action(a, self.player, ateList)
            outcomes.append((score, a))
        action = max(outcomes)[1]
        return action


    def __init__(self, player, evalFn, evalArgs=None):
        super(self.__class__, self).__init__(player)
        self.evaluationFunction = evalFn
        self.evaluationArgs = evalArgs

