import random
from collections import deque

import matplotlib.pyplot as plt

from Environment import Tic_Tac_Toe_State as TTT
from game_utils import playout
import Random_Agent
import Optimal_Agent
import QTable_Agent
import MCTS_Agent
import NeuralNetwork_Agent

def play(env, players, UID, display=True, randomize=False):
	"""Play a single Tic-Tac-Toe game, optionally randomizing player order.

    Args:
        env (Tic_Tac_Toe_State): The Tic-Tac-Toe environment.
        players (list): List of two player agents.
        UID (str): Unique identifier of the player to track.
        display (bool): If True, print game progress.
        randomize (bool): If True, randomize which player goes first.

    Returns:
        tuple: (win, draw, lose) for the tracked player (1 for win, 1 for draw, 1 for loss, 0 otherwise).
    """
	env.render_mode = display
	env.reset()  # Reset environment to initial state

	# Randomize player order if specified
	r = random.randint(0, 1)
	if randomize:
		a, b = players[0], players[1]
		players[r], players[1 - r] = a, b

	rewards = playout(env, players, display)  # Play the game

	# Distribute rewards to players
	players[r].give_reward(rewards[0])
	players[1-r].give_reward(rewards[1])

	# Return results for the tracked player
	for n in range(2):
		if players[n].UID == UID:
			if rewards[n] == -1:
				return 0, 0, 1
			if rewards[n] == 0.5:
				return 0, 1, 0
			if rewards[n] == 1:
				return 1, 0, 0


if __name__ == '__main__':
	"""Main script to run Tic-Tac-Toe games and evaluate agent performance."""
	#random.seed(2)
	ENV = TTT()  # Initialize Tic-Tac-Toe environment

	# Instantiate different agent types
	Randy = Random_Agent.Random()
	Optimus = Optimal_Agent.Optimal()
	Quill = QTable_Agent.QTable(0.1, 100)
	Monte = MCTS_Agent.MCTS(1000)
	Neura = NeuralNetwork_Agent.RLAgent(input_size=27, num_hidden=0, hidden_size=0, output_size=9, batch_size=16)


	# Select agents to play
	p1 = Neura
	p2 = Optimus
	players = [p1, p2]
	player_to_track = 0  # Index of the player to track
	Randomize = False  # Whether to randomize player order

	names = [players[0].agentType, players[1].agentType]  # Agent type names
	print("Started...")

	# Initialize tracking variables
	num_games = 2_000  # Total games to play
	interval = 100  # Window size for moving average and reporting interval
	history = []  # Store moving average win/draw/loss percentages over the last interval games
	cumulative_history = []  # Store cumulative win/draw/loss percentages
	game_results = deque(maxlen=interval)  # Rolling window of game outcomes (win, draw, lose)
	wins, draws, losses = 0, 0, 0  # Cumulative counters
	w, d, l = 0, 0, 0  # Interval counters

	UID = players[player_to_track].UID  # UID of tracked player

	# Run games and track results
	for n in range(num_games):
		win, draw, lose = play(ENV, players, UID, display=False, randomize=Randomize)
		game_results.append((win, draw, lose))  # Store game outcome
		wins += win
		draws += draw
		losses += lose
		total = n + 1  # Total games played so far

		# Sum wins, draws, and losses over the current window for moving average
		moving_wins = sum(result[0] for result in game_results)
		moving_draws = sum(result[1] for result in game_results)
		moving_losses = sum(result[2] for result in game_results)
		moving_total = len(game_results)  # Number of games in the current window

		# Calculate and store moving average percentages
		moving_averages = [100 * moving_wins / moving_total if moving_total > 0 else 0,
		                   100 * moving_draws / moving_total if moving_total > 0 else 0,
		                   100 * moving_losses / moving_total if moving_total > 0 else 0]
		history.append(moving_averages)

		# Calculate and store cumulative percentages
		cumulative_averages = [100 * wins / total, 100 * draws / total, 100 * losses / total]
		cumulative_history.append(cumulative_averages)

		# Update interval counters
		w += win
		d += draw
		l += lose

		# Print interval statistics
		if (n + 1) % interval == 0:
			print(
				f"Game {n + 1} | Wins %: {100 * w / interval} | Draws %: {100 * d / interval} | Losses %: {100 * l / interval}")
			w, d, l = 0, 0, 0  # Reset interval counters

	# Plot moving average win/draw/loss percentages
	plt.figure()
	plt.plot(history)
	plt.xlabel("Number of Games")
	plt.ylabel(f"Moving Average Percentage ({interval}-game window)")
	plt.savefig("moving_average_accuracy.png")

	# Plot cumulative win/draw/loss percentages
	plt.figure()
	plt.plot(cumulative_history)
	plt.xlabel("Number of Games")
	plt.ylabel("Cumulative Percentage")
	plt.savefig("cumulative_accuracy.png")

	# Print final statistics for the tracked player
	print(f"\n{names[player_to_track]} Win Percentage (Moving Average): {history[-1][0]}%")
	print(f"{names[player_to_track]} Draw Percentage (Moving Average): {history[-1][1]}%")
	print(f"{names[player_to_track]} Loss Percentage (Moving Average): {history[-1][2]}%")
	print(f"\n{names[player_to_track]} Win Percentage (Cumulative): {cumulative_history[-1][0]}%")
	print(f"{names[player_to_track]} Draw Percentage (Cumulative): {cumulative_history[-1][1]}%")
	print(f"{names[player_to_track]} Loss Percentage (Cumulative): {cumulative_history[-1][2]}%")