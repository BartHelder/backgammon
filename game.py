import os, sys
import copy
import numpy as np
import random
import time

try:
    import pygame
except:
    print("No module pygame, use screen drawing")

class Game:

    """
    Object representing a single game. Contains functions to play moves, draw graphics and return winning player.
    """
    TOKENS = ['o', 'x']  # white, black
    LAYOUT = "0-2-o,5-5-x,7-3-x,11-5-o,12-5-x,16-3-o,18-5-o,23-2-x"
    NUMCOLS = 24
    QUAD = 6  # number of faces on the dice
    OFF = 'off'
    ON = 'on'


    def __init__(self, layout=LAYOUT, grid=None, graphics=False, offPieces=None, barPieces=None, numPieces=None, players=None):
        """
        Define a new game object
        """
        self.die = Game.QUAD
        self.layout = layout
        if grid:
            self.grid = copy.deepcopy(grid)
            self.offPieces = copy.deepcopy(offPieces)
            self.barPieces = copy.deepcopy(barPieces)
            self.numPieces = copy.deepcopy(numPieces)
            self.players = players
            return
        self.players = Game.TOKENS
        self.grid = [[] for _ in range(Game.NUMCOLS)]
        self.graphics = graphics
        self.offPieces = {}
        self.barPieces = {}
        self.numPieces = {}
        for t in self.players:
            self.barPieces[t] = []
            self.offPieces[t] = []
            self.numPieces[t] = 0
        self.init = True
        self.roll = None
        self.playernum = 0


    @staticmethod
    def new_game(graphics=False):

        g = Game(graphics=graphics)
        g.reset()
        return g

    def reset(self):
        """
        Resets game to original layout.
        """
        for col in self.layout.split(','):
            loc,num,token = col.split('-')
            self.grid[int(loc)] = [token for _ in range(int(num))]
        for col in self.grid:
            for piece in col:
                self.numPieces[piece] += 1

    def step(self, players):

        """
        Takes a single step in the environment: roll dice, draw graphics, decide on and take action, obtain reward
         """

        roll = self.roll_dice()
        if self.graphics:
            self.draw(roll)
            print("Player %s rolled <%d,%d>." % (players[self.playernum].token, roll[0], roll[1]))

        if self.playernum:  # If black is on turn, flip the board around so black has the right (forward) perspective
            self.reverse()
        moves = self.get_actions(roll, self.players[0], nodups=True) # Find all legal moves for this roll
        move = players[self.playernum].get_action(moves, self) if moves else None
        # print("P", self.playernum, " does ", move)
        if move:
            self.take_action(move, self.players[0])
        if self.playernum:  # If black just took his turn, flip the board back to normal 
            self.reverse()

        if self.graphics:
            self.draw(roll)

        done = self.is_done()
        reward = int(self.is_won(players[0].token))  # Reward is 1 only if player 1 (white) has just won

        # Next player:
        self.playernum = (self.playernum + 1) % 2

        return reward, done

    def play(self, players):

        """
        Plays out a full game from start to the end. Returns winning player
        :param players: list of two players (based on agent.Agent)
        :return: player number of winning player
        """
        done = False
        while not done:
            reward, done = self.step(players)
        return self.winner()


    def roll_dice(self):
        return (random.randint(1, self.die), random.randint(1, self.die))

    def take_action(self, action, token):
        """
        Makes given move for player, assumes move is valid, 
        will remove pieces from play
        """
        ateList = [0]*4
        for i,(s, e) in enumerate(action):
            if s==self.ON:
                piece = self.barPieces[token].pop()
            else:
                piece = self.grid[s].pop()
            if e==self.OFF:
                self.offPieces[token].append(piece)
                continue
            if len(self.grid[e])>0 and self.grid[e][0] != token:
                bar_piece = self.grid[e].pop()
                self.barPieces[bar_piece].append(bar_piece)
                ateList[i] = 1
            self.grid[e].append(piece)
        return ateList

    def undo_action(self, action, player, ateList):
        """
        Reverses given move for player, assumes move is valid, 
        will remove pieces from play
        """
        for i, (s, e) in enumerate(reversed(action)):
            if e==self.OFF:
                piece = self.offPieces[player].pop()
            else:
                piece = self.grid[e].pop()
                if ateList[len(action)-1-i]:
                    bar_piece = self.barPieces[self.opponent(player)].pop()
                    self.grid[e].append(bar_piece)
            if s==self.ON:
                self.barPieces[player].append(piece)
            else:
                self.grid[s].append(piece)

    def get_actions(self,roll,player,nodups=False):
        """
        Get set of all possible move tuples.
        """
        moves = set()
        if nodups:
            start=0
        else:
            start=None

        r1,r2 = roll
        if r1 == r2: # doubles
            i = 4
            # keep trying until we find some moves
            while not moves and i>0:
                self.find_moves(tuple([r1] * i), player, (), moves, start)
                i -= 1
        else:
            self.find_moves(roll, player, (), moves, start)
            self.find_moves((r2, r1), player, (), moves, start)
            # has no moves, try moving only one piece
            if not moves:
                for r in roll:
                    self.find_moves((r,), player, (), moves, start)

        return moves

    def find_moves(self, rs, player, move, moves, start=None):
        """
        Helper function for finding moves
        :param rs:
        :param player:
        :param move:
        :param moves:
        :param start:
        :return:
        """
        if len(rs)==0:
            moves.add(move)
            return
        r, rs = rs[0], rs[1:]
        # see if we can remove a piece from the bar
        if self.barPieces[player]:
            if self.can_onboard(player,r):
                piece = self.barPieces[player].pop()
                bar_piece = None
                if len(self.grid[r-1])==1 and self.grid[r-1][-1]!=player:
                    bar_piece = self.grid[r-1].pop()

                self.grid[r-1].append(piece)

                self.find_moves(rs, player, move + ((self.ON, r - 1),), moves, start)
                self.grid[r-1].pop()
                self.barPieces[player].append(piece)
                if bar_piece:
                    self.grid[r-1].append(bar_piece)

            return

        # otherwise check each grid location for valid move using r
        offboarding = self.can_offboard(player)

        for i in range(len(self.grid)):
            if start is not None:
                start = i
            if self.is_valid_move(i,i+r,player):

                piece = self.grid[i].pop()
                bar_piece = None
                if len(self.grid[i+r])==1 and self.grid[i+r][-1]!=player:
                    bar_piece = self.grid[i+r].pop()
                self.grid[i+r].append(piece)
                self.find_moves(rs, player, move + ((i, i + r),), moves, start)
                self.grid[i+r].pop()
                self.grid[i].append(piece)
                if bar_piece:
                    self.grid[i+r].append(bar_piece)

            # If we can't move on the board can we take the piece off?
            if offboarding and self.remove_piece(player,i,r):
                piece = self.grid[i].pop()
                self.offPieces[player].append(piece)
                self.find_moves(rs, player, move + ((i, self.OFF),), moves, start)
                self.offPieces[player].pop()
                self.grid[i].append(piece)

    def opponent(self, token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t!= token: return t

    def is_won(self, player):
        """
        If game is over and player won, return True, else return False
        """
        return self.is_done() and player == self.players[self.winner()]

    def is_lost(self, player):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_done() and player != self.players[self.winner()]

    def reverse(self):
        """
        Reverses a game allowing it to be seen by the opponent
        from the same perspective
        """
        self.grid.reverse()
        self.players.reverse()

    def winner(self):
        """
        Get player number of winning player
        """
        return int(not len(self.offPieces[self.players[0]])==self.numPieces[self.players[0]])

    def is_done(self):
        """
        Checks if the game is done.
        """
        for t in self.players:
            if len(self.offPieces[t])==self.numPieces[t]:
                return True
        return False

    def can_offboard(self,player):
        count = 0
        for i in range(self.NUMCOLS-self.die,self.NUMCOLS):
            if len(self.grid[i])>0 and self.grid[i][0]==player:
                count += len(self.grid[i])
        if count+len(self.offPieces[player]) == self.numPieces[player]:
            return True
        return False

    def can_onboard(self,player,r):
        """
        Can we take a players piece on the bar to a position
        on the grid given by roll-1?
        """
        if len(self.grid[r-1])<=1 or self.grid[r-1][0]==player:
            return True
        else:
            return False

    def remove_piece(self,player,start,r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. no pieces on the bar and all are in the home quadrant.
        """
        if start <self.NUMCOLS - self.die:
            return False
        if len(self.grid[start]) == 0 or self.grid[start][0] != player:
            return False
        if start+r == self.NUMCOLS:
            return True
        if start+r > self.NUMCOLS:
            for i in range(start-1, self.NUMCOLS-self.die-1,-1):
                if len(self.grid[i]) != 0 and self.grid[i][0]==self.players[0]:
                    return False
            return True
        return False

    def is_valid_move(self,start,end,token):
        if len(self.grid[start]) > 0 and self.grid[start][0] == token:
            if end < 0 or end >= len(self.grid):
                return False
            if len(self.grid[end]) <= 1:
                return True
            if len(self.grid[end])>1 and self.grid[end][-1] == token:
                return True
        return False

    def extract_features(self, player, method='original'):
        """
        Transform board state grid into binary feature representation necessary for the ANN.
        :param player: marker of the player that's on turn
        :method: encoding to be used. options:
                    -original (as presented in Tesauro (1995)),
                    -modified (less inputs activated, should train faster?)
        :return: numpy array of 198 binary features
        """

        features = []

        for p in self.players:
            for col in self.grid:
                lc = len(col)
                if lc > 0 and col[0] == p:
                    if method == 'original':
                        feature_map = [[0., 0., 0., 0.], [1., 0., 0., 0.], [1., 1., 0., 0.], [1., 1., 1., 0.]]
                        feats = feature_map[lc] if lc < 4 else [1., 1., 1., (lc - 3) / 2]
                    elif method == 'modified':
                        feature_map = [[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
                        feats = feature_map[lc] if lc < 4 else [0., 0., 1., (lc - 3) / 2]
                    else:
                        raise NotImplementedError('Unknown method specified, exiting')

                else:
                    feats = [0., 0., 0., 0.]
                features += feats
            features.append(len(self.barPieces[p]) / 2)
            features.append(len(self.offPieces[p]) / self.numPieces[p])

        if player == self.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]

        return np.array(features).reshape(1, -1)

    ####################################
    ######## GRAPHICS / DRAWING ########
    ####################################

    def draw(self,roll=None):
        if roll is None:
            roll = self.roll
        else:
            self.roll = roll

        self.draw_gui(roll)

    def init_gui(self):
        pygame.init()
        WIDTH = 800
        HEIGHT = 425
        
        WOFFSET_TOP = 57
        HOFFSET_TOP = 12
        HOFFSET_BOT = 370
        WSKIP = 55
        WMID = 32

        HSKIP = 30
        size = WIDTH,HEIGHT

        self.gridLocs = []

        for i in range(24):
            mid = 0
            hoff = HOFFSET_TOP
            hskip = HSKIP
            k = 11-i
            if i < 6 or i>17:
                mid = WMID
            if i > 11:
                hoff = HOFFSET_BOT
                hskip = -hskip
                k = i-12
            self.gridLocs.append([(WOFFSET_TOP+k*WSKIP+mid,hoff+j*hskip) for j in range(6)])
        self.barLocs = {'x':[(376,142),(376,110)],'o':[(376,243),(376,275)]}
        self.board_img = pygame.transform.scale(pygame.image.load('images/board.png'),size)
        self.screen = pygame.display.set_mode(self.board_img.get_rect().size)
        self.tokIms = {'x':pygame.image.load('images/blackPiece.png'),
                       'o':pygame.image.load('images/whitePiece.png')}
        self.dies = [pygame.transform.scale(pygame.image.load('images/die%d.png'%i),(35,35))
                     for i in range(1,7)]
        self.offIms = {'x':pygame.transform.scale(pygame.image.load('images/blackOff.png'),(40,18)),
                       'o':pygame.transform.scale(pygame.image.load('images/whiteOff.png'),(40,18))}
        
        outOff = 748
        bOffH = 391
        wOffH = 11
        offSkip = 9
        self.offLocs = {'x':[(outOff,bOffH-i*offSkip) for i in range(19)],
                        'o':[(outOff,wOffH+i*offSkip) for i in range(19)]}
        
    def draw_gui(self, roll):
        if self.init:
            self.init_gui()
        self.screen.blit(self.board_img,self.board_img.get_rect())
        self.screen.blit(self.dies[roll[0]-1],(180,190))
        self.screen.blit(self.dies[roll[1]-1],(220,190))
        for i,col in enumerate(self.grid):
            for j,t in enumerate(col):
                # for now only draw first 6 pieces
                if j>5:
                    break
                self.screen.blit(self.tokIms[t],self.gridLocs[23-i][j])
        for k,v in self.barPieces.items():
            for i,t in enumerate(v):
                if i>1:
                    break
                self.screen.blit(self.tokIms[t],self.barLocs[t][i])

        for k,v in self.offPieces.items():
            for i,t in enumerate(v):
                self.screen.blit(self.offIms[t],self.offLocs[t][i])
        pygame.display.flip()

    def gridLocFromPos(self,pos,player):
        tx,ty = self.tokIms['x'].get_rect().size

        def onPiece(pieceLoc,pos,sizex,sizey):
            px,py = pieceLoc
            tx,ty = pos
            if px+sizex > tx > px:
                if ty < py+sizey and ty > py:
                    return True
            return False
            
        # find out if we are on the grid
        for i,col in enumerate(self.grid):
            for loc in self.gridLocs[23-i]:
                if onPiece(loc,pos,tx,ty):
                    return i

        # find out if we are on the bar
        for i,bp in enumerate(self.barPieces[player]):
            if onPiece(self.barLocs[player][i],pos,tx,ty):
                return Game.ON

        # find out if we are removing pieces
        offBase = self.offLocs['o'][0] if player=='o' else self.offLocs['x'][-1]
        offHeight = 200
        offWidth,_ = self.offIms['x'].get_rect().size
        if onPiece(offBase,pos,offWidth,offHeight):
            return Game.OFF
        
        return None
#
# if __name__=='__main__':
#     # g = Game(graphics=True)
#     # g.reset()
#     # players = [RandomAgent(token=Game.TOKENS[0]), RandomAgent(token=Game.TOKENS[1])]
#     #
#     # done = False
#     # while not done:
#     #     reward, done = g.step(players)
