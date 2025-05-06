import copy

import numpy as np

from Environment import Tic_Tac_Toe_State
from Environment import Tic_Tac_Toe_State as new_TTT
import Random_Agent
import Optimal_Agent
import QTable_Agent
import MCTS_Agent
import NeuralNetwork_Agent

import matplotlib.pyplot as plt
import random


def playout(env, players, display=False):

	env.render_mode = display

	observation, rewards, done = env.observation, env.rewards, env.done

	memories = []

	while 1:

		if rewards == [0.5, 0.5]:
			if display:
				print("DRAW\n===========\n\n")
			memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), -1, "ENVIRONMENT", None])
			break

		if done:
			if display:
				print("WIN\n===========\n\n")
			memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), -1, "ENVIRONMENT", None])
			break

		prev_observation = copy.deepcopy(observation)
		state, turn, move_number, _ = observation
		player = players[turn]

		if display: print("{} ({}) to move\n".format(["x","o"][turn], player.agentType))

		action = player.move(observation)
		observation, rewards, done = env.step(action)
		memories.append([copy.deepcopy(prev_observation), copy.deepcopy(observation), done, copy.deepcopy(rewards), turn, player.agentType, action])

	for n in range(len(memories)-1):
		cp_start_obs, cp_new_obs, cp_done, cp_rewards, cp_turn, cp_agent_type, cp_action = memories[n]
		np_start_obs, np_new_obs, np_done, np_rewards, np_turn, np_agent_type, np_action = memories[n+1]
		#final_rewards = memories[-2][3]

		player = players[cp_turn]
		# final_rewards[cp_turn]/((len(memories)-1)/(n+1))

		player.remember(np_done, np_rewards[cp_turn], cp_action, np_new_obs, cp_start_obs)

	return rewards


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
			if rewards[n] == -1:
				return 0, 0, 1
			if rewards[n] == 0.5:
				return 0, 1, 0
			if rewards[n] == 1:
				return 1, 0, 0


if __name__ == '__main__':

	#random.seed(2)

	ENV = new_TTT()

	# Define each agent type
	Randy = Random_Agent.Random()
	Optimus = Optimal_Agent.Optimal()
	Quill = QTable_Agent.QTable(0.1, 100)
	Monte = MCTS_Agent.MCTS(1000)
	Neura = NeuralNetwork_Agent.RLAgent(input_size=27, num_hidden=0, hidden_size=0, output_size=9, batch_size=16)


	# select agents types to player
	p1 = Neura
	p2 = Optimus

	players = [p1, p2]
	player_to_track = 0
	# Randomize which agent goes first each game?
	Randomize = False

	names = [players[0].agentType, players[1].agentType]

	print("Started...")

	history = []
	wins = 0
	w=0
	draws = 0
	d=0
	losses = 0
	l=0

	num_games = 10000
	interval = 500

	UID = players[player_to_track].UID

	for n in range(num_games):

		win, draw, lose = play(ENV, players, UID, display=False, randomize=Randomize)

		wins += win
		w += win
		draws += draw
		d += draw
		losses += lose
		l += lose
		total = wins + draws + losses


		if (n + 1) % interval == 0:
			print(f"Game {n + 1} | Wins %: {100*w/interval} | Draws %: {100*d/interval} | Losses %: {100*l/interval}")
			w, d, l = 0, 0, 0

		averages = [100 * wins / total, 100 * draws / total, 100 * losses / total]
		history.append(averages)

	master = plt.figure()
	plt.plot(history)
	plt.xlabel("Number of Games")
	plt.ylabel("Percentage")
	plt.savefig("cumulative_accuracy.png")

	print("\n{} Win Percentage: {}%".format(names[player_to_track], history[-1][0]))
	print("{} Draw Percentage: {}%".format(names[player_to_track], history[-1][1]))
	print("{} Loss Percentage: {}%".format(names[player_to_track], history[-1][2]))