import pickle

filepath="optimal_moves_dictionary.pkl"

# ---------------------------
# Load from file and display
# ---------------------------

# Load from file
with open(filepath, 'rb') as f:
    loaded_dict = pickle.load(f)

# Extract from dictionary to arrays
states = list(loaded_dict.keys())
moves = list(loaded_dict.values())

# Display
print("Unique States ({}): {}".format(len(states), states))
print("Optimal Moves ({}): {}".format(len(moves), moves))

# ---------------------------------------
# Delete certain amount and save to file
# ---------------------------------------

num = 120  # amount to delete
index = 0  # index to delete

if False:
    for key in states:
        value = loaded_dict[key]

        row = value[1]
        column = value[0]

        new_value = (3 * (int(row) - 1)) + ["c", "b", "a"].index(column)

        Bin = 0b1 << new_value

        loaded_dict[key] = Bin

    with open(filepath, 'wb') as f:
        pickle.dump(loaded_dict, f)

    states = list(loaded_dict.keys())
    moves = list(loaded_dict.values())

    print("Unique States ({}): {}".format(len(states), states))
    print("Optimal Moves ({}): {}".format(len(moves), moves))



if False:
    for key in states:

        new_key = [key[1:10], key[10:19]]

        x = 0b000000000
        for n in range(9):
            if new_key[0][n] == "1":
                x |= 0b1 << 8 - n

        o = 0b000000000
        for n in range(9):
            if new_key[1][n] == "1":
                o |= 0b1 << 8 - n

        new_key = x << 9 | o

        # print("{0:b}".format(new_key))

        loaded_dict[new_key] = loaded_dict.pop(key)

    with open(filepath, 'wb') as f:
        pickle.dump(loaded_dict, f)

    states = list(loaded_dict.keys())
    moves = list(loaded_dict.values())

    print("Unique States ({}): {}".format(len(states), states))
    print("Optimal Moves ({}): {}".format(len(moves), moves))



if False:

    # Delete
    for n in range(num):
        states.pop(index)
        moves.pop(index)

    # Display
    print("Unique States ({}): {}".format(len(states), states))
    print("Optimal Moves ({}): {}".format(len(moves), moves))

    # Compile into dictionary
    optimal_moves = {}

    for n in range(len(states)):
        optimal_moves[states[n]] = moves[n]

    # Save to file
    with open(filepath, 'wb') as f:
        pickle.dump(optimal_moves, f)