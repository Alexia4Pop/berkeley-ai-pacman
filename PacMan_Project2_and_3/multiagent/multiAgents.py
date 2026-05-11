# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if ghostState.scaredTimer == 0 and ghostPos == newPos:
                return -float('inf')

        food_score = 0
        if newFood:
            min_food_distance = min([util.manhattanDistance(newPos, food) for food in newFood])
            food_score = 10.0 / (min_food_distance + 1)

        food_score += (len(currentGameState.getFood().asList()) - len(newFood)) * 100.0

        ghost_effect = 0

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = util.manhattanDistance(newPos, ghostPos)
            scaredTime = ghostState.scaredTimer

            if scaredTime > 0:
                if ghostDist <= scaredTime:
                    ghost_effect += 20.0 / (ghostDist + 1)
            else:
                if ghostDist <= 4:
                    ghost_effect += -100.0 / (ghostDist + 1)

        capsule_bonus = 0
        newCapsules = successorGameState.getCapsules()

        if newCapsules:
            min_capsule_dist = min([util.manhattanDistance(newPos, cap_pos) for cap_pos in newCapsules])

            capsule_bonus = 50.0 / (min_capsule_dist + 1)

        score = successorGameState.getScore()

        final_score = (score * 1.0) + \
                      food_score + \
                      ghost_effect + \
                      capsule_bonus

        if action == 'Stop':
            final_score -= 10.0

        return final_score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        value, action = self.maxValue(gameState, depth=0, agentIndex=0)
        return action

    def isGoalState(self, gameState: GameState, currentDepth: int) -> bool:
        return gameState.isWin() or gameState.isLose() or currentDepth == self.depth

    def maxValue(self, gameState: GameState, depth: int, agentIndex: int):
        if self.isGoalState(gameState, depth):
            return self.evaluationFunction(gameState), None

        v = -float('inf')
        bestAction = None

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            value_from_successor = self.minValue(successorState, depth, 1)

            if value_from_successor > v:
                v = value_from_successor
                bestAction = action

        return v, bestAction

    def minValue(self, gameState: GameState, depth: int, agentIndex: int):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        v = float('inf')
        numAgents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)

            nextAgentIndex = agentIndex + 1
            nextDepth = depth

            if nextAgentIndex == numAgents:
                nextAgentIndex = 0
                nextDepth += 1
                value_from_successor, _ = self.maxValue(successorState, nextDepth, nextAgentIndex)

            else:
                value_from_successor = self.minValue(successorState, nextDepth, nextAgentIndex)

            v = min(v, value_from_successor)

        return v



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        value, action = self.getValue(
            gameState,
            depth=0,
            agentIndex=0,
            alpha=-float('inf'),
            beta=float('inf')
        )
        return action

    def getValue(self, gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None if agentIndex == 0 else self.evaluationFunction(gameState)

        if agentIndex == 0 and depth == self.depth:
            return self.evaluationFunction(gameState), None

        is_pacman_turn = (agentIndex == 0)
        numAgents = gameState.getNumAgents()

        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth + 1 if nextAgentIndex == 0 else depth

        v = -float('inf') if is_pacman_turn else float('inf')
        bestAction = None

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            result = self.getValue(successorState, nextDepth, nextAgentIndex, alpha, beta)
            value_from_successor = result[0] if isinstance(result, tuple) else result

            if is_pacman_turn:
                if value_from_successor > v:
                    v = value_from_successor
                    bestAction = action

                if v > beta:
                    return (v, bestAction) if depth == 0 else v

                alpha = max(alpha, v)

            else:
                v = min(v, value_from_successor)
                if v < alpha:
                    return v

                beta = min(beta, v)

        if is_pacman_turn:
            return v, bestAction
        else:
            return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
