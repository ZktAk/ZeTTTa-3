

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


def get_legal_indices(bitboards):
	x, y, _ = bitboards
	arr = []
	for n in range(9):
		test = (0b1 << n)
		if _ & test == test:
			arr.append(n)
	return arr