
from sample_players import DataPlayer
from isolation import DebugState

import copy
import math
import random

DEBUGMODE = True
DEBUG_PRI = 7

NodeCount = 0

def printDebugMsg(msg, priority=1):
    if DEBUGMODE and priority >= DEBUG_PRI:
        print(msg)

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        methods = ["RANDOM", "MINIMAX", "ALPHABETA_Iterative", "MCTS", "NEGASCOUT", "PVS", "PVS_Iterative", "PVS_ZWS"]
        method = "MCTS"

        printDebugMsg(DebugState(state.board))

        import random
        ply_count_threshold = 2
        # If fewer than ply_count_threshold applied on board, then choose random move.
        if state.ply_count < ply_count_threshold:
            self.queue.put(random.choice(state.actions()))
        else:
            if method == "RANDOM":
                import random
                self.queue.put(random.choice(state.actions()))
            elif method == "MINIMAX":
                # Code from sample_players.py file.
                # return the optimal minimax move at a fixed search depth of 3 plies
                self.queue.put(self.minimax(state, depth=3))
            elif method == "ALPHABETA_Iterative":  # Win ~= 62.5%
                # Alpha-Beta with iterative deepening
                depth_limit = 5
                best_move = None
                for depth in range(1, depth_limit + 1):
                    best_move = self.alpha_beta_search(state, depth)
                printDebugMsg("final best_move = {}".format(best_move))
                print("Alpha Beta Node Count = {}".format(NodeCount))
                self.queue.put(best_move)
            elif method == "MCTS":  # Win ~=
                # Use Monte Carlo Tree Search
                mcts = MCTS_Search(computational_budget=100)
                self.queue.put(mcts.uctSearch(state))
                print("MCTS self.nodeSelectedCtr = {}".format(mcts.nodeSelectedCtr))
                print("MCTS self.nodeExpandedCtr = {}".format(mcts.nodeExpandedCtr))
                print("MCTS self.nodeSimulatedCtr = {}".format(mcts.nodeSimulatedCtr))
                print("MCTS self.nodeBackpropCtr = {}".format(mcts.nodeBackpropCtr))
                print()
            elif method == "NEGASCOUT":  # Win ~= 18%
                # Use NegaScout
                self.queue.put(self.negaScout(state, depth=5))
            elif method == "PVS":  # Win ~= 11.5%
                # Use Principal Variation Search
                self.queue.put(self.principal_variation_search(state, depth=3))
            elif method == "PVS_Iterative":  # Win ~=
                # Use principal variation search with iterative deepening.
                depth_limit = 5
                best_move = None
                for depth in range(1, depth_limit + 1):
                    best_move = self.principal_variation_search(state, depth)
                self.queue.put(best_move)
            elif method == "PVS_ZWS":  # Win ~=
                # Use Principal Variation Search
                self.queue.put(self.principal_variation_search_zws(state, depth=5))
            else:
                import sys
                sys.exit("Unknown method")

        printDebugMsg("self.queue = {}".format(self.queue))


    def negaScout(self, state, depth):
        """ Based on pseudo-code from https://www.researchgate.net/figure/NegaScout-Algorithm-Pseudo-Code-Using-the-Minimal-Window-Search-Principle_fig5_262672371

        INPUTS:
            "state": game state.
            "depth": depth in search tree."""
        def ns(state, depth, alpha, beta):
            if (depth <= 0):
                returnVal = self.score(state)
                printDebugMsg("Reached depth limit. returnVal = {}".format(returnVal))
                return returnVal

            if (state.terminal_test()):
                printDebugMsg("terminal_test()")
                returnVal = state.utility(self.player_id)
                printDebugMsg("Reached terminal limit. returnVal = {}".format(returnVal))
                return returnVal

            score = float("-inf")
            n = beta

            for a in state.actions():
                # Call other player and switch sign of returned value.
                cur = -ns(state, depth-1, -n, -alpha)

                if cur > score:
                    if (n == beta) or (depth <= 2):
                        # compare returned value and score value.
                        # update if necessary.
                        score = cur
                    else:
                        score = -ns(state, depth-1, -beta, -cur)

                # adjust the search window
                alpha = max(alpha, score)

                if alpha >= beta:
                    # Beta cut-off
                    return alpha

                n = alpha + 1

            return score

        alpha = float("-inf")
        beta = float("inf")
        return max(state.actions(), key=lambda x: ns(state.result(x), depth, alpha, beta))



    def principal_variation_search(self, state, depth):
        """Based on pseudo-code from https://en.wikipedia.org/wiki/Principal_variation_search.

        Principal Variation Search is a negamax algorithm that can be faster than alpha-beta pruning.
        Negamax is a variant of minimax that relies on fact that max(a,b) = - min(-a, -b) to simplify
        the minimax algorithm.  The value of a position to player A is the negation of the value to player B.

        Principal Variation Search is sometimes referred to as NegaScout, which is a practically identical algorithm.
        NegaScout is a directional search algorithm to compute the minimax value of a node in a tree.
        It relies on accurate node ordering to have advantage over alpha-beta by never examining a node that can
        be pruned by alpha-beta.

        It assumes the first explored node is the best (i.e. first node is the principal variation).
        Variation refers to specific sequence of successive moves in a turn-based game.
        Principal variation is defined as most advantageous to the current player.

        It then checks whether the first node is the principal variation by searching the remaining nodes with
        a "null window" when alpha and beta are equal.  This is faster than searching with a regular alpha-beta window.
        If this check fails, then the first node was not the principal variation and the search continues with normal
        alpha-beta pruning.

        With good node ordering, Principal Variation Search is faster than alpha-beta pruning.
        If the ordering is bad, then Principal Variation Search can take more time than regula alpha-beta
        due to having to re-search many nodes.

        INPUTS:
            "state": game state.
            "depth": depth in search tree.
        """
        def pvs(state, depth, alpha, beta, player_id):
            """ Function that does the work.
            INPUTS:
                "state": game state.
                "depth": depth in search tree.
                "alpha": alpha value.
                "beta": beta value.
                "player_id": ID of current player. 0/1
            """
            printDebugMsg("\n###pvs: depth={}, alpha={}, beta={}, player_id={}".format(depth, alpha, beta, player_id), priority=7)
            # Terminal test is more important than depth check. If both are true, we want the utility
            # returned rather than the score.
            if (state.terminal_test()):
                printDebugMsg("terminal_test()")
                returnVal = state.utility(player_id)
                printDebugMsg("Reached terminal limit. returnVal = {}".format(returnVal), priority=7)
                return returnVal

            printDebugMsg("\nplayer_id = {}".format(player_id))
            if (depth <= 0):
                returnVal = self.score(state)
                printDebugMsg("Reached depth limit. returnVal = {}".format(returnVal), priority=7)
                return returnVal

            firstChild = True
            printDebugMsg("%%%%%%%%%%%% state.actions = {}".format(state.actions()), priority=7)
            for a in state.actions():
                # printDebugMsg("depth = {}".format(depth), priority=4)
                printDebugMsg("***depth = {} action = {}".format(depth, a), priority=7)
                tmp_ActionOrder_dict[depth].add(a)
                printDebugMsg(tmp_ActionOrder_dict[depth], priority=3)
                if depth == 4:
                    printDebugMsg("action = {}".format(a), priority=4)
                # Initialize score to -infinity
                score = float("-inf")
                if firstChild:
                    printDebugMsg("first child")
                    firstChild = False
                    score = -pvs(state.result(a), depth-1, -beta, -alpha, 1-player_id)
                else:
                    printDebugMsg("NOT first child")
                    # Search with a null window.
                    # Null window has a width 1.
                    printDebugMsg("Search with null window", priority=7)
                    score = -pvs(state.result(a), depth-1, -alpha-1, -alpha, 1-player_id)
                    if alpha < score < beta:
                        # If score is below alpha then the move is worse than we already have, so we can ignore.
                        # If score is above beta, the move is too good to play, so ignore.
                        printDebugMsg("Need to do full re-search. {} < {} < {}".format(alpha, score, beta), priority=7)
                        # If it failed, do a full re-search.
                        score = -pvs(state.result(a), depth-1, -beta, -score, 1-player_id)
                printDebugMsg("alpha = max of alpha {} and score {}".format(alpha, score), priority=7)
                alpha = max(alpha, score)

                # printDebugMsg("score = {}".format(score), priority=7)

                if alpha >= beta:
                    # Beta cut-off
                    printDebugMsg("Beta cut-off.  alpha = {}, beta = {}".format(alpha, beta), priority=7)
                    break
            printDebugMsg("Returning alpha {}".format(alpha), priority=7)
            return alpha

        alpha = float("-inf")
        beta = float("inf")
        player_id = self.player_id  # Part of DataPlayer that is inherited.

        from collections import defaultdict
        tmp_ActionOrder_dict = defaultdict(set)
        returnVal = max(state.actions(), key=lambda x: pvs(state.result(x), depth, alpha, beta, player_id))
        printDebugMsg("Unsorted tmp_ActionOrder_dict = {}".format(tmp_ActionOrder_dict), priority=4)
        # for k, v in sorted(tmp_ActionOrder_dict):
        #     printDebugMsg("depth {} has actions {}".format(k, v), priority=2)
        # import sys
        # sys.exit()
        return returnVal
        return max(state.actions(), key=lambda x: pvs(state.result(x), depth, alpha, beta, player_id))

    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def principal_variation_search_zws(self, state, depth):
        """Based on pseudo-code from https://www.chessprogramming.org/Principal_Variation_Search.

        Principal Variation Search is a negamax algorithm that can be faster than alpha-beta pruning.
        Negamax is a variant of minimax that relies on fact that max(a,b) = - min(-a, -b) to simplify
        the minimax algorithm.  The value of a position to player A is the negation of the value to player B.

        Principal Variation Search is sometimes referred to as NegaScout, which is a practically identical algorithm.
        NegaScout is a directional search algorithm to compute the minimax value of a node in a tree.
        It relies on accurate node ordering to have advantage over alpha-beta by never examining a node that can
        be pruned by alpha-beta.

        It assumes the first explored node is the best (i.e. first node is the principal variation).
        Variation refers to specific sequence of successive moves in a turn-based game.
        Principal variation is defined as most advantageous to the current player.

        It then checks whether the first node is the principal variation by searching the remaining nodes with
        a "null window" when alpha and beta are equal.  This is faster than searching with a regular alpha-beta window.
        If this check fails, then the first node was not the principal variation and the search continues with normal
        alpha-beta pruning.

        With good node ordering, Principal Variation Search is faster than alpha-beta pruning.
        If the ordering is bad, then Principal Variation Search can take more time than regula alpha-beta
        due to having to re-search many nodes.

        INPUTS:
            "state": game state.
            "depth": depth in search tree.
        """
        def pvs(state, depth, alpha, beta):
            """ Function that does the work.
            INPUTS:
                "state": game state.
                "depth": depth in search tree.
                "alpha": alpha value.
                "beta": beta value.
            """
            printDebugMsg("\nplayer_id = {}".format(self.player_id))
            if (depth <= 0):
                returnVal = self.score(state)
                printDebugMsg("Reached depth limit. returnVal = {}".format(returnVal))
                return returnVal

            if (state.terminal_test()):
                printDebugMsg("terminal_test()")
                returnVal = state.utility(self.player_id)
                printDebugMsg("Reached terminal limit. returnVal = {}".format(returnVal))
                return returnVal

            bSearchPv = True
            for a in state.actions():
                if bSearchPv:
                    score = -pvs(state.result(a), depth-1, -beta, -alpha)
                else:
                    score = -zwSearch(state.result(a), depth-1, -alpha)
                    if alpha < score < beta:
                        # Re-search
                        score = -pvs(state.result(a), depth-1, -beta, -alpha)

                if score >= beta:
                    # Fail-hard beta-cutoff
                    return beta
                if score > alpha:
                    # alpha acts like max in Minimax
                    alpha = score
                bSearchPv = False

            return alpha

        def zwSearch(state, beta, depth):
            """ Based on pseudo-code from https://www.chessprogramming.org/Principal_Variation_Search.
            Fail-hard zero window search, returns either beta-1 or beta."""
            if depth <= 0:
                returnVal = self.score(state)
                printDebugMsg("Reached depth limit. returnVal = {}".format(returnVal))
                return returnVal

            for a in state.actions():
                score = -zwSearch(state.result(a), 1-beta, depth-1)

                if score >= beta:
                    # fail-hard beta-cutoff
                    return beta

            # Fail-hard, return alpha
            return beta-1


        alpha = float("-inf")
        beta = float("inf")
        return max(state.actions(), key=lambda x: pvs(state.result(x), depth, alpha, beta))

    def alpha_beta_search(self, state, depth):
        """ Alpha-Beta with Iterative Deepening to use as baseline.
        """
        def ab_min_value(state, alpha, beta, depth):
            """ Return the value for a win (+1) if the game is over,
            otherwise return the minimum value over all legal child
            nodes.
            """
            global NodeCount
            NodeCount += 1
            printDebugMsg("ab_min_value with depth = {}".format(depth))

            if state.terminal_test():
                return state.utility(self.player_id)
            if (depth <= 0):
                return self.score(state)

            v = float("inf")
            for a in state.actions():
                v = min(v, ab_max_value(state.result(a), alpha, beta, depth-1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        def ab_max_value(state, alpha, beta, depth):
            """ Return the value for a loss (-1) if the game is over,
            otherwise return the maximum value over all legal child
            nodes.
            """
            global NodeCount
            NodeCount += 1
            printDebugMsg("ab_max_value with depth = {}".format(depth))
            if state.terminal_test():
                return state.utility(self.player_id)
            if (depth <= 0):
                return self.score(state)

            v = float("-inf")
            for a in state.actions():
                v = max(v, ab_min_value(state.result(a), alpha, beta, depth-1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            printDebugMsg("action = {}".format(a))
            v = ab_min_value(state.result(a), alpha, beta, depth)
            printDebugMsg("v = {}".format(v))
            alpha = max(alpha, v)
            printDebugMsg("alpha = {}".format(alpha))
            printDebugMsg("v > best_score? v = {} best_score = {}".format(v, best_score))
            if v > best_score:
                printDebugMsg("v > best_score")
                best_score = v
                best_move = a
        printDebugMsg("best_move = {}".format(best_move))
        return best_move



    def minimax(self, state, depth):
        """ FOR TESTING FROM sample_players.py file """
        def min_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1))
            return value

        def max_value(state, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1))
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

class MCTS_Node:
    """Monte Carlo Tree Search Node"""
    def __init__(self, state, parent=None):
        """Initialize node.

        INPUTS:
            "state": state of the current node.
            "parent": parent of current node.  Default is none since we initialize root.
        """
        self.state = state
        self.parent = parent

        # Keep track of the total reward. This is updated when we use backupNegamax.
        self.reward = 0.0

        # Need to keep track of children and their actions.
        self.child_list = []
        self.child_actions = []

        # Keep track of number of times each node is visited.
        self.numberTimesVisited = 1

    def isTerminal(self):
        """Check if node is terminal."""
        return self.state.terminal_test()

    def isNonTerminal(self):
        """Algorithm requires verifying node is non-terminal."""
        return not self.isTerminal()

    def isFullyExpanded(self):
        """We are fully expanded when the number of actions we have explored
        is equal to the number of actions available from the current state."""
        return len(self.child_actions) == len(self.state.actions())

    def isNotFullyExpanded(self):
        """Algorithm requires verifying node is not fully expanded."""
        return not self.isFullyExpanded()

    def update(self, delta):
        """Used in the BackupNegamax to update the count and the total rewards.

        INPUTS:
            "delta": delta amount to increment to reward.
        """
        self.reward += delta
        self.numberTimesVisited += 1


    def addChild(self, childState, action):
        """Add child (with state and action) to children of current node.

        INPUTS:
            "childState": state of new child.
            "action": action to reach new child.
        OUTPUTS:
            Return child that was created.
        """
        # We use "self" to refer to our current node.
        newChild = MCTS_Node(childState, self)
        self.child_list.append(newChild)
        self.child_actions.append(action)
        return newChild



class MCTS_Search:
    """Monte Carlo Tree Search Algorithm"""
    def __init__(self, computational_budget):
        """Initialize the search.

        INPUTS:
            "computational_budget": number of iterations through the tree search.
        OUTPUT:
            action returned from uctSearch.
        """
        self.computational_budget = computational_budget
        self.nodeSelectedCtr = 0
        self.nodeExpandedCtr = 0
        self.nodeSimulatedCtr = 0
        self.nodeBackpropCtr = 0

    def uctSearch(self, state):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.
        This is the main method that is used.

        UCT = "Upper Confidence Bound" applied to Trees.

        INPUTS:
            "state": starting state.
        OUTPUT:
            action to be taken.
        """
        # Create root node with state.
        rootNode = MCTS_Node(state)
        # Continue to iterate until we are out of "budget" (i.e. number of times
        # we want to iterate through the tree search).
        for _ in range(self.computational_budget):
            childNode = self.treePolicy(rootNode)
            delta = self.defaultPolicy(childNode.state)
            self.backupNegamax(childNode, delta)
        # Need to get the index of one of the best children and return the action
        # taken to reach it.
        childIndex = rootNode.child_list.index(self.bestChild(rootNode))
        return rootNode.child_actions[childIndex]

    def treePolicy(self, v):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.

        Used for Selection.

        INPUTS:
            "v": node to evaluate.
        OUTPUT:
            returns a node.
        """
        while v.isNonTerminal():
            self.nodeSelectedCtr += 1
            if v.isNotFullyExpanded():
                return self.expand(v)
            else:
                v = self.bestChild(v)

        printDebugMsg("")
        return v

    def expand(self, v):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.

        Used for Expansion.

        Used when we reach the frontier.

        INPUTS:
            "v": node to expand.
        OUTPUT:
            returns a new node child.
        """
        self.nodeExpandedCtr += 1
        tried_actions = set(v.child_actions)
        valid_actions = set(v.state.actions())
        untried_actions = valid_actions - tried_actions

        if len(untried_actions) == 0:
            raise ValueError("untried_actions should not be length 0")

        # Select random action from list of untried actions.
        action = random.choice(list(untried_actions))
        # action = untried_actions[0]  # Select first untried action.
        childState = v.state.result(action)
        newChild = v.addChild(childState, action)
        return newChild

    def bestChild(self, v, c=0.5):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.

        "bestChild() function chooses the action _a_ that maximizes _Q_ over the child nodes
        _v'_ of the input node _v_, so the value of Q should be higher if taking action
        _a_ from state _v_ will lead to the player with iniative in state _v_ (the parent)
        winning from state _v'_ (the child), and lower if taking the action will lead
        to a loss." (from Lesson 5).

        In the pseudo-code Q(v') is defined as the reward of v' and N(v') is the
        number of times v' has been visited.

        INPUTS:
            "v": node to start with.
            "c": tunable parameter to scale the importance of the exploration portion.
        OUTPUT:
            argmax a child node of v.
        """
        # Initialize best score to -infinity.
        bestScore = float("-inf")
        # Keep track of all children that have the best score.
        bestChildren = []

        # Go through all children of v.
        for child in v.child_list:
            # Exploitation portion:
            exploitation = (child.reward / child.numberTimesVisited)
            # math.log is "ln"
            exploration = math.sqrt((2.0 * math.log(v.numberTimesVisited)) / child.numberTimesVisited)
            score = exploitation + c * exploration
            if score == bestScore:
                bestChildren.append(child)
            elif score > bestScore:
                bestChildren = []
                bestChildren.append(child)
                bestScore = score

        # We have gone through all the children here. Now select one of the best children to return.
        return random.choice(bestChildren)

    def defaultPolicy(self, state):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.

        Used for Simulation (i.e. Playout Phase).

        Randomly search the descendants of state and return the reward for state.

        INPUTS:
            "state": state to search descendants of.
        OUTPUT:
            return reward for state.
        """
        # need to perform deep copy of state to prevent overwriting existing state.
        # Used at the end to return the reward.
        startingState = copy.deepcopy(state)

        # Loop through states until reach a terminal state.
        while not state.terminal_test():
            self.nodeSimulatedCtr += 1
            random_action = random.choice(state.actions())
            state = state.result(random_action)

        # Default policy should return +1 if the agent holding initiative at the start of
        # a simulation loses and -1 if the active agent when the simulation starts wins
        # because nodes store the reward relative to their parent in the game tree.

        # Therefore, need to check if the liberties in the terminal state for the
        # player in the starting state wins or loses.
        if state._has_liberties(startingState.player()):
            # Starting state player wins, so need to return -1.
            return -1
        else:
            # Starting state player loses, so need to return +1.
            return 1


    def backupNegamax(self, v, delta):
        """Based on Pseudo-code from Lesson 5 additional adversarial search topics.

        Used for Backpropagation.

        INPUTS:
            "v": Starting node to begin Backpropagation on.
            "delta": delta value of the starting node.
        """
        while v is not None:
            self.nodeBackpropCtr += 1
            v.update(delta)
            delta *= -1
            v = v.parent
