from util import manhattanDistance
from game import Directions
import random, util
import queue

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
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

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    return MiniMax(gameState, self.depth, self.index)[1]
    # END_YOUR_CODE

def MiniMax(gameState, depth, agent):
  if depth == 0 or gameState.isWin() or gameState.isLose():
    return [gameState.getScore(), Directions.STOP]
  if agent == 0:
    maxEval = -9999999
    for action in gameState.getLegalActions(0):
      currEval = MiniMax(gameState.generateSuccessor(0, action), depth - 1, 1)[0]
      if currEval > maxEval:
        maxEval = currEval
        winningAction = action
    return [maxEval, winningAction]
  else:
    minEval = 9999999
    for i in range(gameState.getNumAgents() - 1):
      for action in gameState.getLegalActions(i+1):
        currEval = MiniMax(gameState.generateSuccessor(i+1, action), depth, 0)[0]
        if currEval < minEval:
          minEval = currEval
          winningAction = action
    return [minEval, winningAction]

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    alpha = -99999
    beta = 99999
    return MiniMaxAlpha(gameState, self.depth, alpha, beta, self.index)[1]

def MiniMaxAlpha(gameState, depth, alpha, beta, agent):
  if depth == 0 or gameState.isWin() or gameState.isLose():
    return [gameState.getScore(), Directions.STOP]
  if agent == 0:
    maxEval = -9999999
    for action in gameState.getLegalActions(0):
      currEval = MiniMaxAlpha(gameState.generateSuccessor(0, action), depth - 1, alpha, beta, 1)[0]
      if currEval > maxEval:
        maxEval = currEval
        winningAction = action
        alpha = max(alpha, currEval)
        if beta <= alpha:
          break
    return [maxEval, winningAction]
  else:
    minEval = 9999999
    for i in range(gameState.getNumAgents() - 1):
      for action in gameState.getLegalActions(i+1):
        currEval = MiniMaxAlpha(gameState.generateSuccessor(i+1, action), depth, alpha, beta, 0)[0]
        if currEval < minEval:
          minEval = currEval
          winningAction = action
          beta = min(beta, currEval)
          if beta <= alpha:
            break
    return [minEval, winningAction]
    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    return ExpectiMax(gameState, self.depth, self.index)[1]

def ExpectiMax(gameState, depth, agent):
  if depth == 0 or gameState.isWin() or gameState.isLose():
    return [gameState.getScore(), Directions.STOP]
  if agent == 0:
    maxEval = -9999999
    for action in gameState.getLegalActions(0):
      currEval = ExpectiMax(gameState.generateSuccessor(0, action), depth - 1, 1)[0]
      if currEval > maxEval:
        maxEval = currEval
        winningAction = action
    return [maxEval, winningAction]
  else:
    minEval = 9999999
    for i in range(gameState.getNumAgents() - 1):
      actions = gameState.getLegalActions(i+1)
      action = random.choice(actions)
      currEval = ExpectiMax(gameState.generateSuccessor(i+1, action), depth, 0)[0]
      if currEval < minEval:
        minEval = currEval
        winningAction = action
    return [minEval, winningAction]    
    # END_YOUR_CODE

######################################################################################
# Problem 4a : creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
      Your extreme, unstoppable evaluation function (problem 4).

      DESCRIPTION: minimax function with changed 
      motivation when ghosts are further away.
    """

    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    pacman = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostPositions()
    close = False
    for ghost in ghosts:
      if [abs(pacman[0] - ghost[0]) < 3] and [abs(pacman[1] - ghost[1]) < 3]:
        print("spoooooky")
        close = True
        break
    if close: 
      return MiniMax(currentGameState, 2, 0)[1]
    else: 
      maxEval = -9999999
      for action in gameState.getLegalActions(0):
        currEval = MiniMax(gameState.generateSuccessor(0, action), depth - 1, 1)[0]
        if currEval > maxEval:
          maxEval = currEval
          winningAction = action
      return winningAction
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction

# Problem 4b : Describe your evaluation function.
"""
I wanted my evaluation function to work very similarly to the minimax function except when there are no ghosts within x moves, move towards the nearest piece of food.
The high level motivation is still to stay alive, but when the ghosts are further away the motivation should change. 
"""