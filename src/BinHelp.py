

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
