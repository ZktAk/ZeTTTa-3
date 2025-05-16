import copy


def playout(env, players, display=False):
	"""Simulate a single Tic-Tac-Toe game between two players.

	Args:
	   env (Tic_Tac_Toe_State): The Tic-Tac-Toe environment.
	   players (list): List of two player agents.
	   display (bool): If True, print game progress and results.

	Returns:
	   list: Final rewards for both players.
	"""
	env.render_mode = display  # Set rendering mode
	observation, rewards, done = env.observation, env.rewards, env.done
	memories = []  # Store game states for player learning
	prev_observation = None  # Initialize to None for cases where game ends immediately

	while True:
		# Handle draw case
		if rewards == [0.5, 0.5]:
			if display:
				print("DRAW\n===========\n\n")
			memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), -1, "ENVIRONMENT", None])
			break

		# Handle win/loss case
		if done:
			if display:
				print("WIN\n===========\n\n")
			memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), -1, "ENVIRONMENT", None])
			break

		prev_observation = copy.deepcopy(observation)  # Save current state
		state, turn, move_number, _ = observation  # Unpack observation
		player = players[turn]  # Select current player

		if display:
			print("{} ({}) to move\n".format(["x","o"][turn], player.agentType))

		action = player.move(observation)  # Get player's move
		observation, rewards, done = env.step(action)  # Update environment
		memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), turn, player.agentType, action])

	game_result = memories[-1][3]
	# Update player memory with game outcomes
	for n in range(len(memories)-1):
		cp_start_obs, cp_new_obs, cp_done, cp_rewards, cp_turn, cp_agent_type, cp_action = memories[n]
		np_start_obs, np_new_obs, np_done, np_rewards, np_turn, np_agent_type, np_action = memories[n+1]
		#final_rewards = memories[-2][3]

		player = players[cp_turn]
		# final_rewards[cp_turn]/((len(memories)-1)/(n+1))

		player.remember(np_done, np_rewards[cp_turn], cp_action, np_new_obs, cp_start_obs, game_result[cp_turn])

	return rewards


def pretty_bin(integer, n):
	"""Format a binary number as a string with leading zeros and underscores for readability.

    Args:
        integer (int): The integer to convert to a binary string.
        n (int): The desired length of the binary string (including leading zeros).

    Returns:
        str: A formatted binary string prefixed with '0b', with underscores every three digits.
    """
	str = "{0:b}".format(integer)  # Convert integer to binary string

	# Add leading zeros to reach desired length
	while len(str) < n:
		str = "0" + str

	# Reverse the string for underscore insertion
	reversed_s = str[::-1]

	# Insert underscores every three characters
	parts = [reversed_s[i:i + 3] for i in range(0, len(reversed_s), 3)]
	underscored_reversed_s = '_'.join(parts)

	# Reverse back to original order
	str = underscored_reversed_s[::-1]

	return "0b" + str


def get_legal_indices(bitboards):
	"""Identify legal move indices from the Tic-Tac-Toe bitboards.

	Args:
	 bitboards (list): List of three bitboards [x, y, empty] representing the game state.

	Returns:
	 list: List of indices (0-8) corresponding to empty squares where a move is legal.
	"""
	x, y, _ = bitboards  # Unpack bitboards
	arr = []
	for n in range(9):
		test = (0b1 << n)  # Create a bitmask for position n
		if _ & test == test:  # Check if the position is empty
			arr.append(n)
	return arr


def get_bit(number, bit):
	"""Extract the value of a specific bit from a number.

	Args:
	    number (int): The number to extract the bit from.
	    bit (int): The position of the bit (0-based).

	Returns:
	    int: The value of the bit (0 or 1).
	"""
	return (number >> bit) & 1


def bin_to_array(bin, length):
	"""Convert a binary number to an array of its bits.

    Args:
        bin (int): The binary number to convert.
        length (int): The desired length of the output array.

    Returns:
        list: List of bits (0s and 1s) representing the binary number.
    """
	array = []

	for n in range(length):
		bit = get_bit(bin, n)  # Extract each bit
		array.append(bit)

	return array
