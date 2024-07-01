import copy


def pretty_bin(integer, n):
	str = "{0:b}".format(integer)

	while len(str) < n:
		str = "0" + str

	# Reverse the string
	reversed_s = str[::-1]

	# Insert underscore after every third character
	parts = [reversed_s[i:i + 3] for i in range(0, len(reversed_s), 3)]
	underscored_reversed_s = '_'.join(parts)

	# Reverse the string back to its original order
	str = underscored_reversed_s[::-1]

	return "0b" + str


class Tic_Tac_Toe_State:
	def __init__(self):
		self.piece_dict = {0b00: " ", 0b10: "x", 0b01: "o", 0b11: "ERROR: Illegal Piece State"}
		self.winPositions = [0b111000000, 0b000111000, 0b000000111,
		                     0b100100100, 0b010010010, 0b001001001,
		                     0b100010001, 0b001010100]
		self.x_bitboard= None
		self.y_bitboard = None
		self.draw = False
		self.win = False
		self.reset()

	def reset(self):
		self.x_bitboard = 0b000000000
		self.y_bitboard = 0b000000000
		self.draw = False
		self.win = False

	def isWin(self):
		for pos in self.winPositions:
			if self.x_bitboard & pos == pos: return True
			if self.y_bitboard & pos == pos: return True
		return False

	def isDraw(self):
		if not self.isWin():
			if (~(self.x_bitboard | self.y_bitboard) & 0b111111111) == 0: return True
		return False

	def isTerminal(self):
		return self.isWin() or self.isDraw()

	def update(self):
		self.win = self.isWin()
		self.draw = self.isDraw()

	def getLegalMoves(self, format="array"):
		legalSpace = ~self.x_bitboard & ~self.y_bitboard
		legalMoves = []

		legal_bitboard = 0b0

		for n in range(9):
			test = (0b1 << n)
			if legalSpace & test == test:
				legal_bitboard |= 0b1 << n
				legalMoves.append(test)

		if format == "array": return legalMoves
		if format == "bitboard": return legal_bitboard

	def getLegalStates(self, x, y):
		env = Tic_Tac_Toe_State()
		env.move(x, 0)
		env.move(y, 1)
		legalStates = []
		legalMoves = self.getLegalMoves()

		exes = 0
		for n in range(9):
			bit = (self.x_bitboard & (0b1 << n)) >> n
			if bit == 0b1:
				exes += 1
		oes = 0
		for n in range(9):
			bit = (self.y_bitboard & (0b1 << n)) >> n
			if bit == 0b1:
				oes += 1

		for move in legalMoves:
			if exes == oes:
				new_env = copy.deepcopy(env)
				new_env.move(move, 0)
				legalStates.append([move, new_env, new_env.x_bitboard << 9 | new_env.y_bitboard])
			else:
				new_env = copy.deepcopy(env)
				new_env.move(move, 1)
				legalStates.append([move, new_env, new_env.x_bitboard << 9 | new_env.y_bitboard])

		return legalStates


	def move(self, sqaure, piece):
		if sqaure & (self.x_bitboard | self.y_bitboard) > 0:
			print("ERROR: Tic_Tac_Toe_State.move(): \'Square {} is already occupied\'".format(pretty_bin(sqaure, 9)))
			return

		if piece == 0:
			self.x_bitboard = self.x_bitboard | sqaure
		elif piece == 1:
			self.y_bitboard = self.y_bitboard | sqaure
		else:
			print("ERROR: Tic_Tac_Toe_State.move(): \'piece other than 1 or 0 called\'".format(pretty_bin(sqaure, 9)))
			return

		self.update()


	def get_bit(self, number, bit):
		return (number >> bit) & 1


	def get_piece(self, bit):
		return self.piece_dict[(self.get_bit(self.x_bitboard, bit) << 1) + self.get_bit(self.y_bitboard, bit)]


	def print_board(self):
		str = ""
		for cell in range(9):
			str += " {} ".format(self.get_piece(8 - cell))
			if not ((1 + cell) % 3 == 0): str += "|"
			else:
				print(str)
				str = ""
				if not (cell % 8 == 0):	print("-----------")
		print("\n")


# Below is debug code

"""
state = Tic_Tac_Toe_State()

state.print_board()
state.move(0b101010010, 0)
state.move(0b010101101, 1)


if state.isWin():
	print("WIN")
if state.isDraw():
	print("DRAW")

state.print_board()
state.reset()

if state.isWin():
	print("WIN")
if state.isDraw():
	print("DRAW")

state.reset()
state.move(0b001000010, 0)
state.move(0b000101101, 1)

state.print_board()

legalMoves =  state.getLegalMoves()


for move in legalMoves:
	print("0b{0:b}".format(move))
"""