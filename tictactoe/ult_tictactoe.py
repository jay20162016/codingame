import numpy as np

save = True
load = True 


def check(board):
    for b in [board, [list(x) for x in np.transpose(board) ], list(reversed(board))]:
        if [1, 1, 1] in b:
            return 1
        if [-1, -1, -1] in b:import numpy as np

save = True
load = True


def check(board):
    for b in [board, [list(x) for x in np.transpose(board) ], list(reversed(board))]:
        if [1, 1, 1] in b:
            return 1
        if [-1, -1, -1] in b:
            return -1
        if b[0][0] == b[1][1] == b[2][2] == -1:
            return -1
        if b[0][0] == b[1][1] == b[2][2] == 1:
            return 1
    for b in board:
        if 0 in b:
            return 0
    return -999


class Board:
    def __init__(self, board=None):
        if board is None:
            board = [[0 for i in range(3)] for j in range(3)]
        self.board = board
        self.iterindex = 0
        self.turn = 0

    def show(self, board = None):
        if not board:
            b = self.board
        else:
            b = board
        for sb in b:
            print( ''.join( [str(tile) for tile in sb] ).replace("-1", "O").replace("1", "X").replace("0", "-") )

    def __int__(self):
        return check(self.board)

    def __iter__(self):
        self.iterindex = 0
        return self

    def __next__(self):
        x = self.iterindex
        self.iterindex += 1
        if self.iterindex > 3:
            raise StopIteration
        return self.board[x]

    def __getitem__(self, item):
        if type(item) == tuple:
            return self.board[item[0]][item[1]]
        else:
            return self.board[item]

    def __setitem__(self, key, value):
        if type(key) == tuple:
            self.board[key[0]][key[1]] = value
        else:
            self.board[key] = value

    def __call__(self, *pos):
        if len(pos) == 1:
            pos = pos[0]
        if self[pos] == 0:
            self[pos] = (self.turn % 2) * 2 - 1
        else:
            raise ValueError("Invalid Position")

        self.turn += 1

    def __str__(self):
        nb = ""
        for sb in self.board:
            for tile in sb:
                nb += str(tile + 1)
        return str(int(nb, 3))

    @staticmethod
    def fromstr(s):
        b3 = np.base_repr(int(s),base=3)
        return Board( [ [ int(b) - 1 for b in b3[a : a + 3] ] for a in range(0, 9, 3) ] )


def expand(board):
    expanded = []
    for r in range(9):
        subboard = []
        for sb in board[r // 3]:
            subboard += sb[r % 3]
        expanded.append(subboard)
    return expanded


def sliceboard(board, bx, ex, by, ey):
    xsboard = board[bx:ex]
    ysboard = np.transpose(xsboard)[by:ey]
    return Board( [ list(i) for i in list( np.transpose(ysboard) ) ] )


def compress(board):
    return [[sliceboard(board, 0, 3, 0, 3), sliceboard(board, 0, 3, 3, 6), sliceboard(board, 0, 3, 6, 9)],
            [sliceboard(board, 3, 6, 0, 3), sliceboard(board, 3, 6, 3, 6), sliceboard(board, 3, 6, 6, 9)],
            [sliceboard(board, 6, 9, 0, 3), sliceboard(board, 6, 9, 3, 6), sliceboard(board, 6, 9, 6, 9)]]


class UltBoard(Board):
    def __init__(self, board=None):
        if board is None:
            board = [[Board() for i in range(3)] for j in range(3)]
        super(UltBoard, self).__init__(board)

    def show(self):
        super(UltBoard, self).show(expand(self.board))

    def __next__(self):
        x = self.iterindex
        self.iterindex += 1
        if self.iterindex > 9:
            raise StopIteration
        return expand(self.board)[x]

    def __getitem__(self, item):
        if type(item) == tuple:
            return expand(self.board)[item[0]][item[1]]
        else:
            return expand(self.board)[item]

    def __setitem__(self, key, value):
        if type(key) == tuple:
            ex = expand(self.board)
            ex[key[0]][key[1]] = value
            cmp = compress(ex)
            self.board = cmp
        else:
            ex = expand(self.board)
            ex[key] = value
            cmp = compress(ex)
            self.board = cmp

    def __int__(self):
        bl = []
        for sb in self.board:
            bl.append([])
            for mb in sb:
                bl[-1].append(int(mb))
        return check(bl)

    def __str__(self):
        nb = ""
        for sb in expand(self.board):
            for tile in sb:
                nb += str(tile + 1)
        return str(int(nb, 3))

    @staticmethod
    def fromstr(s):
        b3 = np.base_repr(int(s),base=3)
        return UltBoard( compress( [ [ int(b) - 1 for b in b3[a : a + 9] ] for a in range(0, 81, 9) ] ) )


class BaseBot:
    def __call__(self, board):
        return

    def update(self, board, action, reward, nboard):
        pass


import pandas as pd
from copy import deepcopy as copy
import random


class QLearningBot(BaseBot):
    def __init__(self, gamma, epsilon, bsize, Q = None):
        if Q is None:
            self.Q = pd.DataFrame(columns = [ "(" + str(a) + ", " + str(b) + ")" for a in range(bsize) for b in range(bsize)] ).astype("float")
        else:
            self.Q = Q
        self.gamma = gamma
        self.epsilon = epsilon
        self.bsize = bsize

    def __call__(self, board, rplay = False):
        if rplay:
            return self.Q.loc[int(str(board))].idxmax()
        if random.random() < self.epsilon and int(str(board)) in self.Q.T.columns and \
                not pd.isnull(self.Q.loc[ int(str(board)) ].idxmax()):
            # print("EXPLOIT")
            return self.Q.loc[ int(str(board)) ].idxmax()
        else:
            # print("EXPLORE")
            vactions = []
            for i, x in enumerate(board):
                for j, y in enumerate(x):
                    if y == 0:
                        vactions.append((i, j))
            return random.choice(vactions)

    def update(self, board, action, reward, nboard):
        if not int(str(board)) in self.Q.T.columns:
            self.Q.loc[ int(str(board)) ] = [np.nan for i in range(3**2)]

        if int(str(nboard)) in self.Q.T.columns and not pd.isnull(self.Q.loc[ int(str(nboard)) ].max()):
            self.Q.loc[ int(str(board)) ][str(action)] = reward + self.gamma * self.Q.loc[ int(str(nboard)) ].max()
        else:
            self.Q.loc[ int(str(board)) ][str(action)] = reward


def train(num = np.inf):
    if load:
        the_bot = QLearningBot(-0.8, 0.3, 3, Q = pd.read_csv("ttc.csv", index_col=0))
    else:
        the_bot = QLearningBot(-0.8, 0.3, 3)
    games = 0
    while games < num:
        board = Board()
        while not int(board):
            reward = -1
            pboard = copy(board)
            act = the_bot(board)
            if type(act) == str:
                act = eval(act)
            board(act)
            if int(board) == -((board.turn % 2) * 2 - 1):  # Negate as the turn was already inverted
                reward = 100
            if int(board) == -999:
                reward = 10
            the_bot.update(pboard, act, reward, board)
        games += 1
        if games % 300 == 0:
            # print(games, the_bot.Q.shape[0], the_bot.Q.shape[0] * the_bot.Q.shape[1] - the_bot.Q.isnull().sum().sum())
            print("Games: {0}, # of States: {1},  Filled In Q Size: {2}". \
                  format(games, the_bot.Q.shape[0], the_bot.Q.shape[0] * the_bot.Q.shape[1] - the_bot.Q.isnull().sum().sum()))
            if save:
                the_bot.Q.to_csv("ttc.csv")


def play():
    if load:
        the_bot = QLearningBot(-0.8, 0.3, 3, Q = pd.read_csv("ttc.csv", index_col=0))
    else:
        the_bot = QLearningBot(-0.8, 0.3, 3)
    board = Board()
    while not int(board):
        act = the_bot(board, True)
        if type(act) == str:
            act = eval(act)
        board(act)
        board.show()
        print("=" * 3)
    print("Winner: {}".format({-999: "None", -1: "O", 1: "X"}[int(board)]))


class Board:
    def __init__(self, board=None):
        if board is None:
            board = [[0 for i in range(3)] for j in range(3)]
        self.board = board
        self.iterindex = 0
        self.turn = 0

    def show(self, board = None):
        if not board:
            b = self.board
        else:
            b = board
        for sb in b:
            print( ''.join( [str(tile) for tile in sb] ).replace("-1", "O").replace("1", "X").replace("0", "-") )

    def __int__(self):
        return check(self.board)

    def __iter__(self):
        self.iterindex = 0
        return self

    def __next__(self):
        x = self.iterindex
        self.iterindex += 1
        if self.iterindex > 3:
            raise StopIteration
        return self.board[x]

    def __getitem__(self, item):
        if type(item) == tuple:
            return self.board[item[0]][item[1]]
        else:
            return self.board[item]

    def __setitem__(self, key, value):
        if type(key) == tuple:
            self.board[key[0]][key[1]] = value
        else:
            self.board[key] = value

    def __call__(self, *pos):
        if len(pos) == 1:
            pos = pos[0]
        if self[pos] == 0:
            self[pos] = (self.turn % 2) * 2 - 1
        else:
            raise ValueError("Invalid Position")

        self.turn += 1

    def __str__(self):
        nb = ""
        for sb in self.board:
            for tile in sb:
                nb += str(tile + 1)
        return str(int(nb, 3))

    @staticmethod
    def fromstr(s):
        b3 = np.base_repr(int(s),base=3)
        return Board( [ [ int(b) - 1 for b in b3[a : a + 3] ] for a in range(0, 9, 3) ] )


def expand(board):
    expanded = []
    for r in range(9):
        subboard = []
        for sb in board[r // 3]:
            subboard += sb[r % 3]
        expanded.append(subboard)
    return expanded


def sliceboard(board, bx, ex, by, ey):
    xsboard = board[bx:ex]
    ysboard = np.transpose(xsboard)[by:ey]
    return Board( [ list(i) for i in list( np.transpose(ysboard) ) ] )


def compress(board):
    return [[sliceboard(board, 0, 3, 0, 3), sliceboard(board, 0, 3, 3, 6), sliceboard(board, 0, 3, 6, 9)],
            [sliceboard(board, 3, 6, 0, 3), sliceboard(board, 3, 6, 3, 6), sliceboard(board, 3, 6, 6, 9)],
            [sliceboard(board, 6, 9, 0, 3), sliceboard(board, 6, 9, 3, 6), sliceboard(board, 6, 9, 6, 9)]]


class UltBoard(Board):
    def __init__(self, board=None):
        if board is None:
            board = [[Board() for i in range(3)] for j in range(3)]
        super(UltBoard, self).__init__(board)

    def show(self):
        super(UltBoard, self).show(expand(self.board))

    def __next__(self):
        x = self.iterindex
        self.iterindex += 1
        if self.iterindex > 9:
            raise StopIteration
        return expand(self.board)[x]

    def __getitem__(self, item):
        if type(item) == tuple:
            return expand(self.board)[item[0]][item[1]]
        else:
            return expand(self.board)[item]

    def __setitem__(self, key, value):
        if type(key) == tuple:
            ex = expand(self.board)
            ex[key[0]][key[1]] = value
            cmp = compress(ex)
            self.board = cmp
        else:
            ex = expand(self.board)
            ex[key] = value
            cmp = compress(ex)
            self.board = cmp

    def __int__(self):
        bl = []
        for sb in self.board:
            bl.append([])
            for mb in sb:
                bl[-1].append(int(mb))
        return check(bl)

    def __str__(self):
        nb = ""
        for sb in expand(self.board):
            for tile in sb:
                nb += str(tile + 1)
        return str(int(nb, 3))

    @staticmethod
    def fromstr(s):
        b3 = np.base_repr(int(s),base=3)
        return UltBoard( compress( [ [ int(b) - 1 for b in b3[a : a + 9] ] for a in range(0, 81, 9) ] ) )


class BaseBot:
    def __call__(self, board):
        return

    def update(self, board, action, reward, nboard):
        pass


import pandas as pd
from copy import deepcopy as copy
import random


class QLearningBot(BaseBot):
    def __init__(self, gamma, epsilon, bsize, Q = None):
        if Q is None:
            self.Q = pd.DataFrame(columns = [ "(" + str(a) + ", " + str(b) + ")" for a in range(bsize) for b in range(bsize)] ).astype("float")
        else:
            self.Q = Q
        self.gamma = gamma
        self.epsilon = epsilon
        self.bsize = bsize

    def __call__(self, board, rplay = False):
        if rplay:
            return self.Q.loc[int(str(board))].idxmax()
        if random.random() < self.epsilon and int(str(board)) in self.Q.T.columns and \
                not pd.isnull(self.Q.loc[ int(str(board)) ].idxmax()):
            # print("EXPLOIT")
            return self.Q.loc[ int(str(board)) ].idxmax()
        else:
            # print("EXPLORE")
            vactions = []
            for i, x in enumerate(board):
                for j, y in enumerate(x):
                    if y == 0:
                        vactions.append((i, j))
            return random.choice(vactions)

    def update(self, board, action, reward, nboard):
        if not int(str(board)) in self.Q.T.columns:
            self.Q.loc[ int(str(board)) ] = [np.nan for i in range(3**2)]

        if int(str(nboard)) in self.Q.T.columns and not pd.isnull(self.Q.loc[ int(str(nboard)) ].max()):
            self.Q.loc[ int(str(board)) ][str(action)] = reward + self.gamma * self.Q.loc[ int(str(nboard)) ].max()
        else:
            self.Q.loc[ int(str(board)) ][str(action)] = reward


def train(num = np.inf):
    if load:
        the_bot = QLearningBot(-0.8, 0.3, 3, Q = pd.read_csv("ttc.csv", index_col=0))
    else:
        the_bot = QLearningBot(-0.8, 0.3, 3)
    games = 0
    while games < num:
        board = Board()
        while not int(board):
            reward = -1
            pboard = copy(board)
            act = the_bot(board)
            if type(act) == str:
                act = eval(act)
            board(act)
            if int(board) == -((board.turn % 2) * 2 - 1):  # Negate as the turn was already inverted
                reward = 100
            if int(board) == -999:
                reward = 10
            the_bot.update(pboard, act, reward, board)
        games += 1
        if games % 300 == 0:
            # print(games, the_bot.Q.shape[0], the_bot.Q.shape[0] * the_bot.Q.shape[1] - the_bot.Q.isnull().sum().sum())
            print("Games: {0}, # of States: {1},  Filled In Q Size: {2}". \
                  format(games, the_bot.Q.shape[0], the_bot.Q.shape[0] * the_bot.Q.shape[1] - the_bot.Q.isnull().sum().sum()))
            if save:
                the_bot.Q.to_csv("ttc.csv")


def play():
    if load:
        the_bot = QLearningBot(-0.8, 0.3, 3, Q = pd.read_csv("ttc.csv", index_col=0))
    else:
        the_bot = QLearningBot(-0.8, 0.3, 3)
    board = Board()
    while not int(board):
        act = the_bot(board, True)
        if type(act) == str:
            act = eval(act)
        board(act)
        board.show()
        print("=" * 3)
    print("Winner: {}".format({-999: "None", -1: "O", 1: "X"}[int(board)]))


def playhuman(first = None):
    if first is None:
        first = random.choice([True, False])

    if load:
        the_bot = QLearningBot(-0.8, 0.3, 3, Q = pd.read_csv("ttc.csv", index_col=0))
    else:
        the_bot = QLearningBot(-0.8, 0.3, 3)
    board = Board()
    while not int(board):
        if bool(board.turn % 2) == first:
            act = the_bot(board, True)
        else:
            act = input("Where do you want to go? ")
        if type(act) == str:
            act = eval(act)
        board(act)
        board.show()
        print("=" * 3)