# This is the human-aided program that was used create
# a dictionary of all the optimal Tic-Tac-Toe moves
# to be used by the 'Optimal' agent in Agents.py.

# The program asks the user to manually enter the
# optimal move for each unique state it encounters.

# The optimal moves were based off of this infographic: https://xkcd.com/832/
# which was originally found linked in this StackExchange post:
# https://puzzling.stackexchange.com/a/47

import Environments
import pickle


class OptimalTrainer:

    def __init__(self, filepath="optimal_moves_dictionary.pkl"):
        self.states = []  # list of all states that have been previously visited
        self.moves = []  # list of optimal move for all states
        self.filepath = filepath

        self.load_from_file()

        self.env = Environments.TicTacToeState()
        self.acceptable_input = ["a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3",
                                 "A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    def print(self):
        print("Unique States ({}): {}".format(len(self.states), self.states))
        print("Optimal Moves ({}): {}".format(len(self.moves), self.moves))

    def load_from_file(self):
        with open(self.filepath, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.states = list(loaded_dict.keys())
        self.moves = list(loaded_dict.values())

    def save_to_file(self):
        optimal_moves = {}

        for n in range(len(self.states)):
            optimal_moves[self.states[n]] = self.moves[n]

        with open(self.filepath, 'wb') as f:
            pickle.dump(optimal_moves, f)

    def solve_for_x(self, env):
        self.explore_state(env)

    def solve_for_o(self, env):
        current_state = env
        possible_actions = current_state.getPossibleActions()
        for action in possible_actions:
            new_state = current_state.takeAction(action)
            self.explore_state(new_state)  # recursion

    def solve(self):
        self.solve_for_x(self.env)
        self.save_to_file()
        self.solve_for_o(self.env)
        self.save_to_file()

    def explore_state(self, env):  # this is a recursive function

        current_state = env

        if not current_state.isTerminal():

            action = None

            if current_state.get_bitboard()[2:-1] in self.states:  # If this unique state has already been
                # encountered, then pull optimal move from list instead of asking the user.
                index = self.states.index(current_state.get_bitboard()[2:-1])
                move = self.moves[index]

                row = 3 - (int(move[-1]) - 1)
                column = ["a", "b", "c"].index(move[0].lower()) + 1

                action = current_state.action(row, column)

            else:  # If this unique state has not already been encountered, then ask the user to enter the optimal move.
                current_state.print(2)

                player_to_move = current_state.getPossibleActions()[0].player
                player_to_move = "x" if player_to_move == 1 else "o"

                print(str(player_to_move) + " to move.")

                while True:

                    move = input("Please select the optimal move.")
                    while move not in self.acceptable_input:  # check if the input is valid
                        move = input("Invalid Input! Please select the optimal move (\'a1\', \'B3\', \'c2\').")

                    row = 3 - (int(move[-1]) - 1)
                    column = ["a", "b", "c"].index(move[0].lower()) + 1

                    action = current_state.action(row, column)

                    if action.ID not in current_state.getActsForCompare():  # check if the move is legal
                        print("Invalid Move!")
                    else:
                        break

                self.states.append(current_state.get_bitboard()[2:-1])
                self.moves.append(move)
                self.save_to_file()

            current_state = current_state.takeAction(action)
            possible_actions = current_state.getPossibleActions()

            for act in possible_actions:
                new_state = current_state.takeAction(act)
                self.explore_state(new_state)  # recursion


Optimus = OptimalTrainer()
# Optimus.print()
Optimus.solve()
