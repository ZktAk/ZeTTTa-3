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