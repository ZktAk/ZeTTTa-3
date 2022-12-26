import numpy as np

class Env():

	def __init__(self, x, y, symbols):

		self.rows = x
		self.columns = y
		self.symbols = symbols
		self.state = []

		for n in range(len(symbols)-1):
			self.state.append(np.zeros((x,y), int))

		self.state.append(np.ones((x,y), int))
		self.state = np.array(self.state)



class TicTacToe(Env):

	def __init__(self):
		super().__init__(3, 3, symbols=["x","o"," "])

		self.winPositions = np.array(["0b11110000001", "0b10001110001", "0b10000001111",
		                              "0b11001001001", "0b10100100101", "0b10010010011",
		                              "0b11000100011", "0b10010101001"])

	def move(self, type, row, column):

		if self.legal(row, column):
			self.state[type][row][column] = 1
			self.state[-1][row][column] = 0

		else:
			print("Specified Move ({},{}) Is Not Legal!".format(row, column))

		"""win, draw, winner = self.win()
		if win:
			print("{} Won!".format(self.symbols[winner]))
			return True
		elif draw:
			print("Game is a Draw")
			return False

		return None"""


	def legal(self, row, column):
		return self.state[-1][row][column] == 1


	def bitboard(self, array):
		bitboard = "0b1"

		for row in array:
			for val in row:
				bitboard += str(val)

		bitboard += "1"

		return bitboard


	def win(self):

		for piece in range(len(self.state)-1):
			bitboard = self.bitboard(self.state[piece])  # Convert the 2D array to a 1D bitboard
			for pos in self.winPositions:
				if bin(int(bitboard, 2) & int(pos, 2)) == bin(int(pos, 2)):
					return True, False, piece

		if self.state[-1].sum() == 0:
			return False, True, None

		return False, False, None


	def print(self, spaces):

		for n in range(spaces):
			print("")

		for r in range(self.rows):
			str = ""
			for c in range(self.columns):
				for p in range(len(self.state)):
					if self.state[p][r][c] == 1:
						str += self.symbols[p] + " | "

			str = str[0:-3]
			print(str)
			if (r != self.rows-1):
				print("---------")


if False:
	ttt = TicTacToe()
	ttt.move(0,0,0)
	ttt.move(1,0,1)
	ttt.move(0,0,2)
	ttt.move(1,1,0)
	ttt.move(0,1,1)
	ttt.move(1,1,2)
	ttt.move(1,2,0)
	ttt.move(0,2,1)
	ttt.move(1,2,2)

	ttt.move(1,2,2)

	ttt.print(2)

	ttt.__init__()

	ttt.move(1,2,2)

	ttt.print(2)



