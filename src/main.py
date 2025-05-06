import copy
import random
import matplotlib.pyplot as plt

from Environment import Tic_Tac_Toe_State as TTT
import Random_Agent
import Optimal_Agent
import QTable_Agent
import MCTS_Agent
import NeuralNetwork_Agent


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

	# Update player memory with game outcomes
	for n in range(len(memories)-1):
		cp_start_obs, cp_new_obs, cp_done, cp_rewards, cp_turn, cp_agent_type, cp_action = memories[n]
		np_start_obs, np_new_obs, np_done, np_rewards, np_turn, np_agent_type, np_action = memories[n+1]
		#final_rewards = memories[-2][3]

		player = players[cp_turn]
		# final_rewards[cp_turn]/((len(memories)-1)/(n+1))

		player.remember(np_done, np_rewards[cp_turn], cp_action, np_new_obs, cp_start_obs)

	return rewards


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
	history = []  # Store win/draw/loss percentages
	wins, draws, losses = 0, 0, 0
	w, d, l = 0, 0, 0  # Interval counters
	num_games = 10000  # Total games to play
	interval = 500  # Reporting interval
	UID = players[player_to_track].UID  # UID of tracked player

	# Run games and track results
	for n in range(num_games):
		win, draw, lose = play(ENV, players, UID, display=False, randomize=Randomize)
		wins += win
		w += win
		draws += draw
		d += draw
		losses += lose
		l += lose
		total = wins + draws + losses

		# Print interval statistics
		if (n + 1) % interval == 0:
			print(f"Game {n + 1} | Wins %: {100*w/interval} | Draws %: {100*d/interval} | Losses %: {100*l/interval}")
			w, d, l = 0, 0, 0  # Reset interval counters

		# Calculate and store cumulative percentages
		averages = [100 * wins / total, 100 * draws / total, 100 * losses / total]
		history.append(averages)

	# Plot cumulative win/draw/loss percentages
	master = plt.figure()
	plt.plot(history)
	plt.xlabel("Number of Games")
	plt.ylabel("Percentage")
	plt.savefig("cumulative_accuracy.png")

	# Print final statistics for the tracked player
	print("\n{} Win Percentage: {}%".format(names[player_to_track], history[-1][0]))
	print("{} Draw Percentage: {}%".format(names[player_to_track], history[-1][1]))
	print("{} Loss Percentage: {}%".format(names[player_to_track], history[-1][2]))