import numpy as np
import Environments
from MCTS import MCTS_NN_Agent, Node


def available_moves(board):
    return np.argwhere(board == 0)


def check_game_end(board):
    best = max(list(board.sum(axis=0)) +    # columns
               list(board.sum(axis=1)) +    # rows
               [board.trace()] +            # main diagonal
               [np.fliplr(board).trace()],  # other diagonal
               key=abs)
    if abs(best) == board.shape[0]:  # assumes square board
        return np.sign(best)  # winning player, +1 or -1
    if available_moves(board).size == 0:
        return 0  # a draw (otherwise, return None by default)



def play(player_objs, Env, display=False):

    Env.__init__()
    node = Node(Env.board)

    player = 1
    game_end = Env.isWin()

    move = 0

    while game_end is None:
        move += 1

        print("\tmove {}".format(move))

        node = player_objs[int((player/2)+0.5)].move(node)
        game_end = check_game_end(node.board)
        player *= -1  # switch players

        if display: Env.print(2)

    return game_end


if __name__ == '__main__':

    TicENV = Environments.TicTacToeState()

    player1 = MCTS_NN_Agent()
    player_objs = [player1, player1]

    num_games = 50

    for n in range(num_games):

        print("Running game {} of {}...".format(n+1, num_games))

        reward = play(player_objs, TicENV)
        player1.gameEnd(abs(reward))


    for n in range(20):
        player1.train_Model()




