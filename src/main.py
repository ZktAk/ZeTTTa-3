from Environment import Tic_Tac_Toe_State
from Environment import Tic_Tac_Toe_State as new_TTT
import Random_Agent
import Optimal_Agent
import QTable_Agent
import MCTS_Agent

import matplotlib.pyplot as plt
import random


def playout(env, players, display=False):

	env.render_mode = display

	observation, rewards, done = env.observation, env.rewards, env.done

	while 1:

		if rewards == [0.5, 0.5]:
			if display:
				print("DRAW\n===========\n\n")
			return rewards

		if done:
			if display:
				print("WIN\n===========\n\n")
			return rewards


		prev_obs = observation
		state, turn, _ = observation
		player = players[turn]

		if display: print("{} ({}) to move\n".format(["x","o"][turn], player.agentType))

		action = player.move(observation)

		observation, rewards, done = env.step(action)

		player.remember(done, action, observation, prev_obs)



def play(env, players, UID, display=True, randomize=False):
	# track: The index of the player whose stats we want to track

	env.render_mode = display
	_ = env.reset()

	r = random.randint(0, 1)

	if randomize:
		a = players[0]
		b = players[1]
		players[r] = a
		players[1-r] = b

	rewards = playout(env, players, display)

	players[r].give_reward(rewards[0])
	players[1-r].give_reward(rewards[1])

	for n in range(2):
		if players[n].UID == UID:
			if rewards[n] == 0:
				return 0, 0, 1
			if rewards[n] == 0.5:
				return 0, 1, 0
			if rewards[n] == 1:
				return 1, 0, 0


if __name__ == '__main__':

	random.seed(1)

	ENV = new_TTT()

	# Define each agent type
	Randy = Random_Agent.Random()
	Optimus = Optimal_Agent.Optimal()
	Quill = QTable_Agent.QTable(0.1, 100)
	Monte = MCTS_Agent.MCTS(1000)

	# select agents types to player
	p1 = Monte
	p2 = Optimus

	players = [p1, p2]
	player_to_track = 0

	# Randomize which agent goes first each game?
	Randomize = True

	names = [players[0].agentType, players[1].agentType]

	print("Started...")

	history = []
	wins = 0
	draws = 0
	losses = 0

	num_games = 100
	interval = 1

	UID = players[player_to_track].UID

	for n in range(num_games):

		if (n + 1) % interval == 0:
			print("Game {}".format(n + 1))


		win, draw, lose = play(ENV, players, UID, display=False, randomize=Randomize)

		wins += win
		draws += draw
		losses += lose

		averages = [100 * wins / (n + 1), 100 * draws / (n + 1), 100 * losses / (n + 1)]
		history.append(averages)

	master = plt.figure()
	plt.plot(history)
	plt.xlabel("Number of Games")
	plt.ylabel("Percentage")
	plt.savefig("cumulative_accuracy.png")

	print("\n{} Win Percentage: {}%".format(names[player_to_track], history[-1][0]))
	print("{} Draw Percentage: {}%".format(names[player_to_track], history[-1][1]))
	print("{} Loss Percentage: {}%".format(names[player_to_track], history[-1][2]))