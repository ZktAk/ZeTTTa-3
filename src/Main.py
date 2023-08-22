import math

import numpy as np
import random
import Environments
import Agents
import matplotlib.pyplot as plt


def randomizePlayers():
    PlayerX = random.randint(0, 1)  # The player with the crosses always goes first.
    PlayerO = abs(PlayerX - 1)  # PlayerO = 1 if PlayerX == 0 else 0

    return PlayerX, PlayerO


def play(env, players, display=False, Random=True):
    current_state = env()  # reset the environment

    if display:
        current_state.print(2)

    if Random:
        PlayerX, PlayerO = randomizePlayers()
    else:
        PlayerX = 0
        PlayerO = 1

    currentPlayer = PlayerX

    while True:

        action = players[currentPlayer].move(current_state)
        current_state = current_state.takeAction(action)

        currentPlayer = PlayerO if currentPlayer == PlayerX else PlayerX

        if display:
            current_state.print(1)

        if current_state.isWin():
            if display: print("{} Won!".format(current_state.symbols[action.symbol]))
            players[currentPlayer].giveReward(-1)

            if currentPlayer == 1:
                players[0].giveReward(1)
            else:
                players[1].giveReward(1)

            return

        elif current_state.isDraw():
            if display: print("Game is a Draw")

            players[0].giveReward(0)
            players[1].giveReward(0)

            return


if __name__ == '__main__':

    TicENV = Environments.TicTacToeState
    players = [Agents.MCTS(), Agents.Optimal()]  # Agents.QTable(0.99083)

    history = []

    print("Started...")

    play(TicENV, [players[1], players[0]], display=False, Random=False)

    for n in range(10):

        if (n + 1) % (1) == 0:
            print("Game {}".format(n + 1))
        play(TicENV, players, display=False)
        history.append(players[0].getAverages())

    play(TicENV, [players[1], players[0]], display=True, Random=False)
    play(TicENV, [players[0], players[1]], display=True, Random=False)


    master = plt.figure()
    plt.plot(history)
    plt.xlabel("Number of Games")
    plt.ylabel("Win Percentage")
    plt.savefig("cumulative_accuracy.png")

    print("Win Percentage: {}%".format(100 * history[-1][0]))
    print("Loss Percentage: {}%".format(100 * history[-1][2]))
    print("Draw Percentage: {}%".format(100 * history[-1][1]))

    #print("unique game scenarios: {} | {}".format(len(players[0].masterNodes),len(players[1].nodes)))

    '''mcts = players[1]
    starting_node = list(mcts.nodes.values())[0]
    visits = []
    scores = []
    UCT = []
    explore = []
    success = []
    for child in starting_node.children:
        visits.append(child.visits)
        scores.append(child.winsOrDraws)
        UCT.append(child.UCT())
        explore.append(child.exploration_rate())
        success.append(child.success_rate())
    print("\nMCTS Initial State Action Visits: {}".format(visits))
    print("MCTS Initial State Action Scores: {}".format(scores))
    print("MCTS Initial State Action explore: {}".format(explore))
    print("MCTS Initial State Action success: {}".format(success))
    print("MCTS Initial State Action UCT: {}\n".format(UCT))'''


    for n in range(10):
        print("Move Rankings for Position {}: {}".format(n, players[0].masterNodes[n].actions.values()))
# print("Move Rankings for Starting Position: {}".format(players[0].masterNodes[2].actions.values()))
