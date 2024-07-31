from BinHelp import pretty_bin


def get_bit(number, bit):
	return (number >> bit) & 1


class Tic_Tac_Toe_State:

	def __init__(self, render_mode=None):

		if render_mode is None:
			self.render_mode = False
		elif render_mode == "human":
			self.render_mode = True
		else:
			print("render_mode other than 'human' given!")

		self.piece_dict = {0b00: " ", 0b10: "x", 0b01: "o", 0b11: "ERROR: Illegal Piece State"}
		self.winPositions = [0b111000000, 0b000111000, 0b000000111,
		                     0b100100100, 0b010010010, 0b001001001,
		                     0b100010001, 0b001010100]

		self.bitboards = None
		self.current_move = None
		self.rewards = None
		self.done = None
		self.observation = None

		self.reset()


	def set_render_mode(self, render_mode):
		# True or False
		self.render_mode = render_mode


	def reset(self):
		self.bitboards = [0b000000000, 0b000000000, 0b111111111]
		self.current_move = 0  # 0 means that it is x's turn to move. 1 means it is o's turn to move
		self.rewards = [0, 0]
		self.done = False

		self.render()

		self.observation = [self.bitboards, self.current_move, self]
		return self.observation


	def set(self, observation):
		self.bitboards = observation[0]
		self.current_move = observation[1]
		self.rewards = [0, 0]

		self.done = self.check_terminal()

		self.render()


	def step(self, index):

		if self.done:
			print("ERROR: Tic_Tac_Toe_State.step(): \'Tried to move when state is already terminal\'")
			return

		sqaure = 0b1 << index

		if sqaure & (self.bitboards[0] | self.bitboards[1]) > 0:
			print("ERROR: Tic_Tac_Toe_State.step(): \'Square {} is already occupied\'".format(pretty_bin(sqaure, 9)))
			return

		self.bitboards[self.current_move] = self.bitboards[self.current_move] | sqaure
		self.bitboards[2] = ~(self.bitboards[0] | self.bitboards[1]) & 0b111111111
		self.current_move = 1 - self.current_move

		self.observation = [self.bitboards, self.current_move, self]
		self.done = self.check_terminal()

		self.render()

		return self.observation, self.rewards, self.done


	def check_terminal(self):

		# check if winning position
		for pos in self.winPositions:

			for piece in range(2):
				if self.bitboards[piece] & pos == pos:
					self.rewards[piece] = 1
					return True

		# check if draw position
		if self.bitboards[2] == 0:
			self.rewards = [0.5, 0.5]
			return True

		return False


	def get_piece(self, bit):
		return self.piece_dict[(get_bit(self.bitboards[0], bit) << 1) + get_bit(self.bitboards[1], bit)]


	def render(self):
		if not self.render_mode: return

		output_string = ""
		for cell in range(9):
			output_string += " {} ".format(self.get_piece(8 - cell))
			if not ((1 + cell) % 3 == 0):
				output_string += "|"
			else:
				print(output_string)
				output_string = ""
				if not (cell % 8 == 0): print("-----------")
		print("\n")
