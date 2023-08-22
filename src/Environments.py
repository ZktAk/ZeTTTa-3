import numpy as np
import math
import copy


def convertIndexTo2D(index, shape=(3, 3)):
    row = math.floor(index / shape[0])
    cell = index - (row * shape[1])
    return row, cell


class Action:
    def __init__(self, player, row, column):
        self.player = player
        self.row = row
        self.column = column

        symbolsDict = {1: 0, -1: 1}
        self.symbol = symbolsDict[player]

        self.ID = "{}:{}:{}".format(self.row, self.column, self.player)


class EnvState:

    def __init__(self, x, y, symbols, State=None):

        self.rows = x
        self.columns = y
        self.symbols = symbols
        self.currentPlayer = 1

        if State is not None:
            self.board = State
        else:
            self.board = []

            for n in range(len(symbols) - 1):
                self.board.append(np.zeros((x, y), int))

            self.board.append(np.ones((x, y), int))
            self.board = np.array(self.board)

    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self):
        possibleActions = []
        for row in range(len(self.board[-1])):
            for column in range(len(self.board[-1][row])):

                # print("Move: [{},{}]".format(row, column))
                if self.board[-1][row][column] == 1:
                    possibleActions.append(Action(self.currentPlayer, row, column))
        return possibleActions

    def getActsForCompare(self):
        actions = self.getPossibleActions()
        IDs = []
        for action in actions:
            IDs.append(action.ID)
        return IDs

    def action(self, row, column):
        return Action(self.currentPlayer, row-1, column-1)

    def takeAction(self, action):
        newState = copy.deepcopy(self)

        # print("Symbol: {}\nAction: [{},{}]".format(action.symbol, action.row, action.column))

        newState.board[action.symbol][action.row][action.column] = 1
        newState.board[-1][action.row][action.column] = 0
        newState.currentPlayer = self.currentPlayer * -1
        return newState

    def isWin(self):
        pass

    def isDraw(self):
        pass

    def isTerminal(self):
        pass

    def getReward(self):
        if self.isWin():  # only give a reward if the game is won
            return self.currentPlayer * -1
        # after the last action was taken, the current player was switched.
        # self.currentPlayer * -1 switches to the previous player, the one that just moved.
        # Thus, we can use the value stored in self.currentPlayer as the reward.
        return False

    def print(self, spaces):

        for n in range(spaces):
            print("")

        print("-------------")

        for r in range(self.rows):
            output = "| "
            for c in range(self.columns):
                for p in range(len(self.board)):
                    if self.board[p][r][c] == 1:
                        output += self.symbols[p] + " | "


            #output = output[0:-3]
            output += str(3-r)
            print(output)
            if r != self.rows - 1:
                print("|-----------| ")
        print("-------------")
        print("  A   B   C")


class TicTacToeState(EnvState):

    def __init__(self, state=None):
        super().__init__(3, 3, symbols=["x", "o", " "], State=state)

        self.winPositions = np.array(["0b11110000001", "0b10001110001", "0b10000001111",
                                      "0b11001001001", "0b10100100101", "0b10010010011",
                                      "0b11000100011", "0b10010101001"])

    def bitboard(self, array):
        bitboard = "0b1"

        for row in array:
            for val in row:
                bitboard += str(val)

        bitboard += "1"
        return bitboard


    def get_bitboard(self):
        bitboard = "0b1"

        for piece in self.board:
            for row in piece:
                for val in row:
                    bitboard += str(val)

        bitboard += "1"
        return bitboard


    def isWin(self):

        for piece in range(len(self.board) - 1):
            bitboard = self.bitboard(self.board[piece])  # Convert the 2D array to a 1D bitboard
            # print("Bitboard of Current State: {}".format(bitboard))
            for pos in self.winPositions:
                if bin(int(bitboard, 2) & int(pos, 2)) == bin(int(pos, 2)):
                    return True
        return False

    def isDraw(self):

        if not self.isWin():
            if self.board[-1].sum() == 0:
                return True
        return False

    def isTerminal(self):
        return self.isDraw() or self.isWin()
