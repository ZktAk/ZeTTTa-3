from game_utils import pretty_bin, get_bit


class Tic_Tac_Toe_State:
	"""A class representing the state of a Tic-Tac-Toe game, using bitboards for efficient state management."""

	def __init__(self, render_mode=None):
		"""Initialize the Tic-Tac-Toe game state.

        Args:
            render_mode (str, optional): Determines if the board is rendered visually. 'human' enables rendering,
            None disables it.
        """

		# Determine whether to print the game board to the console
		if render_mode is None:
			self.render_mode = False
		elif render_mode == "human":
			self.render_mode = True
		else:
			print("render_mode other than 'human' given!")

		# Dictionary mapping bit patterns to piece symbols
		# 0b00: empty, 0b10: 'x', 0b01: 'o', 0b11: illegal state
		self.piece_dict = {
			0b00: " ",
			0b10: "x",
			0b01: "o",
			0b11: "ERROR: Illegal Piece State"
		}

		# Binary masks for all 8 possible win conditions (rows, columns, diagonals)
		self.winPositions = [
			0b111000000, 0b000111000, 0b000000111,  # rows
			0b100100100, 0b010010010, 0b001001001,  # columns
			0b100010001, 0b001010100                # diagonals
		]

		# Initialize game state variables
		self.bitboards = None       # List of bitboards: [x pieces, o pieces, empty squares]
		self.current_move = None    # Tracks whose turn it is (0 for x, 1 for o)
		self.move_counter = None    # Counts total moves made
		self.rewards = None         # Rewards for each player [x, o]
		self.done = None            # Indicates if the game is over
		self.observation = None     # Packaged game state

		# Initialize/reset the game
		self.reset()


	def set_render_mode(self, render_mode):
		"""Enable or disable rendering the board to the console.

        Args:
            render_mode (bool): True to enable rendering, False to disable.
        """
		self.render_mode = render_mode


	def reset(self):
		"""Reset the game to its initial state.

        Returns:
            list: The initial observation of the game state.
        """
		# Initialize bitboards: [x pieces, o pieces, empty squares]
		self.bitboards = [0b000000000, 0b000000000, 0b111111111]
		self.current_move = 0   # x starts (0 for x, 1 for o)
		self.move_counter = 0   # Reset move counter
		self.rewards = [0, 0]   # Reset rewards for both players
		self.done = False       # Game is not terminal

		# Render the initial board if rendering is enabled
		self.render()

		# Update observation with current state
		self.observation = [self.bitboards, self.current_move, self.move_counter, self]
		return self.observation


	def set(self, observation):
		"""Set the game state to a given observation.

        Args:
            observation (list): Contains [bitboards, current_move, move_counter, self].
        """
		# Update state from observation
		self.bitboards = observation[0]
		self.current_move = observation[1]
		self.move_counter = observation[2]
		self.rewards = [0, 0]  # Reset rewards

		# Check if the state is terminal
		self.done = self.check_terminal()

		# Render the board if rendering is enabled
		self.render()


	def step(self, index):
		"""Perform a move at the specified index.

        Args:
            index (int): The position (0-8) where the current player places their piece.

        Returns:
            tuple: (observation, rewards, done) - Current state, rewards, and terminal status.
        """
		# Check if the game is already over
		if self.done:
			print("ERROR: Tic_Tac_Toe_State.step(): \'Tried to move when state is already terminal\'")
			return

		# Create a bitmask for the selected square
		sqaure = 0b1 << index

		# Check if the square is already occupied
		if sqaure & (self.bitboards[0] | self.bitboards[1]) > 0:
			print("ERROR: Tic_Tac_Toe_State.step(): \'Square {} is already occupied\'".format(pretty_bin(sqaure, 9)))
			self.done = True
			self.rewards[self.current_move] = -100  # Penalty for illegal move
			return self.observation, self.rewards, self.done

		# Place the piece in the current player's bitboard
		self.bitboards[self.current_move] = self.bitboards[self.current_move] | sqaure
		# Update empty squares bitboard
		self.bitboards[2] = ~(self.bitboards[0] | self.bitboards[1]) & 0b111111111
		# Switch to the other player's turn
		self.current_move = 1 - self.current_move

		# Check if the game is now terminal
		self.done = self.check_terminal()
		if not self.done:
			self.move_counter += 1  # Increment move counter if game continues

		# Update observation
		self.observation = [self.bitboards, self.current_move, self.move_counter, self]

		# Render the board if rendering is enabled
		self.render()

		return self.observation, self.rewards, self.done


	def check_terminal(self):
		"""Check if the game has reached a terminal state (win or draw).

        Returns:
            bool: True if the game is over, False otherwise.
        """
		# Check for a winning position
		for pos in self.winPositions:
			for piece in range(2):  # Check both players (0 for x, 1 for o)
				if self.bitboards[piece] & pos == pos:
					self.rewards[piece] = 1  # Winner gets +1
					self.rewards[1-piece] = -1  # Loser gets -1
					return True

		# Check for a draw (no empty squares)
		if self.bitboards[2] == 0:
			self.rewards = [0.5, 0.5]  # Both players get 0.5 for a draw
			return True

		return False


	def get_piece(self, bit):
		"""Get the piece at a specific bit position.

        Args:
            bit (int): The bit position (0-8) to check.

        Returns:
            str: The piece symbol (" ", "x", "o", or error message).
        """
		# Combine bits from both bitboards to determine the piece
		return self.piece_dict[(get_bit(self.bitboards[0], bit) << 1) + get_bit(self.bitboards[1], bit)]


	def render(self):
		"""Render the current game board to the console if rendering is enabled."""
		if not self.render_mode:
			return

		output_string = ""
		for cell in range(9):
			# Get the piece at the current position
			output_string += " {} ".format(self.get_piece(8 - cell))
			if not ((1 + cell) % 3 == 0):
				output_string += "|"  # Add column separator
			else:
				print(output_string)  # Print row
				output_string = ""
				if not (cell % 8 == 0):
					print("-----------")  # Add row separator
		print("\n")  # Extra newline for clarity
