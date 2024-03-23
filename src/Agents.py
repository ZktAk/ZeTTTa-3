import copy
import math
import pickle
import random
import numpy as np
from Environments import Action
from Environments import TicTacToeState


def containsElement(arr, val):
    """Check if a value is present in an array."""
    for n in arr:
        if n == val: return True
    return False

def arrEqualsArr(arr1, arr2):
    """Check if two arrays are equal."""
    elementsThatMatch = arr1.flatten() == arr2.flatten()
    for val in elementsThatMatch:
        if not val:
            return False
    return True


def containsArray(superArr, subArr):
    """Check if an array is present in a list of arrays."""
    for arr in superArr:
        if arrEqualsArr(arr, subArr): return True
    return False


def to2DIndex(index1D, shape=(3, 3)):
    """Convert a 1D index to a 2D index."""
    row = math.floor(index1D / shape[0])
    cell = index1D - (row * shape[1])
    return row, cell

class Agent:
    def __init__(self):
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.numGames = 0

        self.prevWins = 0
        self.prevDraws = 0
        self.prevLosses = 0

        self.winHist = [0]
        self.drawHist = [0]
        self.lossHist = [0]

    def give_reward(self, reward):
        self.numGames += 1
        if reward == 1:
            self.wins += 1
        elif reward == 0:
            self.draws += 1
        elif reward == -1:
            self.losses += 1

    def get(self):
        return [self.wins, self.draws, self.losses]

    def getAverages(self):
        winP = self.wins / self.numGames
        drawP = self.draws / self.numGames
        lossP = self.losses / self.numGames

        return [winP, drawP, lossP]

    def backprop(self):
        pass
class Optimal(Agent):
    def __init__(self, path="optimal_moves_dictionary.pkl"):
        super().__init__()
        self.agentType = "Optimal"
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.dictionary = loaded_dict

    def move(self, initialState):
        current_state = initialState
        current_bitboard = current_state.get_bitboard()[2:-1]
        move = self.dictionary[current_bitboard]

        row = 3 - (int(move[-1]) - 1)
        column = ["a", "b", "c"].index(move[0].lower()) + 1

        action = current_state.action(row, column)
        return action


class Random(Agent):
    def __init__(self):
        super().__init__()
        self.agentType = "Random"

    def move(self, initialState):
        possibleActions = initialState.get_possible_actions()
        if len(possibleActions) == 0:
            return None
        return random.choice(possibleActions)


class qTable_Node:
    def __init__(self, initial_state, identifier):
        """Initializes a Q-table node with initial state and identifier."""
        self.state = initial_state
        self.id = identifier
        self.PossibleActions = self.state.get_possible_actions()
        self.actions = {action: 0 for action in self.PossibleActions}


def locate(identifier, arr):
    """Locate a node with the provided identifier in the array."""
    for n, node in enumerate(arr):
        if arrEqualsArr(node.id, identifier):
            return n
    return False

class QTable(Agent):

    def __init__(self, convergence, num_games):
        """Initializes a Q-table agent with convergence and number of games.

        Qtable(self, C, N)

        C	--	convergence

        N	--	number of games

        C^(1/N)	--	yields exploration gamma"""

        super().__init__()
        self.agentType = "Q-Table"
        self.p = 1
        self.gamma = convergence ** (1 / float(num_games))
        self.master_nodes = []
        self.game_nodes = []
        self.game_actions = []

    def move(self, state):
        """Selects an action based on the state."""
        current_state = qTable_Node(state, identifier=np.copy(state.board))
        index = locate(current_state.id, self.master_nodes)
        if index is False:
            self.master_nodes.append(current_state)
        else:
            current_state = self.master_nodes[index]
        self.game_nodes.append(current_state)
        if random.random() <= self.p:
            actions = list(current_state.actions.keys())
        else:
            best_val = float("-inf")
            actions = []
            for action, value in current_state.actions.items():
                if value == best_val:
                    actions.append(action)
                elif value > best_val:
                    best_val = value
                    actions = [action]
        action = random.choice(actions)
        self.game_actions.append(action)
        return action

    def give_reward(self, reward):
        """Assigns rewards to actions taken during the game."""
        super().give_reward(reward)
        score = 0.5 if reward == 0 else reward
        for game_node, action_taken in zip(self.game_nodes, self.game_actions):
            game_node.actions[action_taken] += score
        self.game_nodes = []
        self.game_actions = []
        self.p *= self.gamma


class MCTSNode():

    def __init__(self, state, parent=None, action=None):
        self.state = copy.deepcopy(state)
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def expand(self):
        possible_actions = self.state.get_possible_actions()
        for action in possible_actions:
            new_state = self.state.takeAction(action)
            new_node = MCTSNode(new_state, self, action)
            self.children.append(new_node)

    def isFullyExpanded(self):
        possible_actions = self.state.get_possible_actions()
        return len(self.children) == len(possible_actions)

    def UCB1(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def selectChild(self):
        selected_child = max(self.children, key=lambda child: child.UCB1())
        return selected_child

    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)
class MCTS(Agent):

    def __init__(self, num_simulations=50):
        super().__init__()
        self.agentType = "MCTS"
        self.num_simulations = num_simulations

    def move(self, state):
        root = MCTSNode(state, action=None)
        for _ in range(self.num_simulations):
            node = self.select_node(root)
            result = self.simulate(node)
            node.backpropagate(result)
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def select_node(self, root):
        node = root
        while not node.state.isTerminal():
            if not node.isFullyExpanded():
                node.expand()
                return node.children[0]
            else:
                node = node.selectChild()
        return node

    def simulate(self, node):
        current_state = node.state
        player = 1
        while not current_state.isTerminal():
            player *= -1
            possible_actions = current_state.get_possible_actions()
            action = random.choice(possible_actions)
            current_state = current_state.takeAction(action)
        player *= -1
        # if (player != current_state.getReward()): print("{} != {}".format(player, current_state.getReward()))
        return current_state.getReward() if current_state.getReward() != 0 else 0.9
