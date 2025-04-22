from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)

        """
        terminalState: 
        This function checks if the game has ended or 
        if the search depth limit has been reached. If either condition is met, 
        it returns True; otherwise, it returns False.

        minimaxScore: 
        This function finds the best score for the current game state, 
        considering whether it's Pacman's or the ghosts' turn. 
        It recursively evaluates possible actions to maximize Pacman's score and minimize the ghosts' score.

        """
        
        def terminalState(gameState, depth):
            """
            Check if the game is over or the depth limit is reached.

            Args:
                gameState: The current game state.
                depth: The current depth of the search tree.

            Returns:
                bool: True if the game is over or the depth limit is reached, False otherwise.
            """
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def minimaxScore(agentState, depth, gameState):
            """
            Calculate the minimax score for the current state.

            Args:
                agentState: The index of the current agent.
                depth: The current depth of the search tree.
                gameState: The current game state.

            Returns:
                float: The minimax score for the current state.
            """
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            if agentState == 0:  # Pacman's turn
                maxScore = float("-inf")
                for action in gameState.getLegalActions(agentState):
                    maxScore = max(maxScore, minimaxScore(1, depth, gameState.getNextState(agentState, action)))
                return maxScore
            else:  # Ghost's turn
                nextAgentState = (agentState + 1) % gameState.getNumAgents()
                if nextAgentState == 0:
                    nextDepth = depth + 1
                else:
                    nextDepth = depth
                minScore = float("inf")
                for action in gameState.getLegalActions(agentState):
                    minScore = min(minScore, minimaxScore(nextAgentState, nextDepth, gameState.getNextState(agentState, action)))
                return minScore

        bestScore = float("-inf")
        bestAction = None

        # Iterate over Pacman's legal actions
        for action in gameState.getLegalActions(0):
            score = minimaxScore(1, 0, gameState.getNextState(0, action))
            # Update best score and action if a better score is found
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        # End your code (Part 1)





class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)

        def terminalState(gameState, depth):
            """
            Check if the game is over or the depth limit is reached.

            Args:
                gameState: The current game state.
                depth: The current depth of the search tree.

            Returns:
                bool: True if the game is over or the depth limit is reached, False otherwise.
            """
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def alpha_beta_search(agentState, depth, gameState, alpha, beta):
            """
            Perform alpha-beta pruning search to find the best action.

            Args:
                agentState: The index of the current agent.
                depth: The current depth of the search tree.
                gameState: The current game state.
                alpha: The current alpha value.
                beta: The current beta value.

            Returns:
                float: The score of the best action.
            """
            # Check whether the game has ended or the defined depth is reached
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)
            
            # Get the next agent state and update the depth when all agents have been traversed
            nextAgentState = (agentState + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgentState == 0 else depth
                    
            # If the agent is Pacman, find the maximum score for all legal actions of the agent and prune unnecessary branches
            if agentState == 0:
                maximumScore = float("-inf")
                for action in gameState.getLegalActions(agentState):
                    # Find the maximum score of Pacman
                    score = alpha_beta_search(nextAgentState, nextDepth, gameState.getNextState(agentState, action), alpha, beta)
                    # Prune the branch
                    if score > beta:
                        return score
                    # Update alpha
                    alpha = max(alpha, score)
                    maximumScore = max(maximumScore, score)
                return maximumScore
            # If the agent is a ghost, find the minimum score for all legal actions of the agent and prune unnecessary branches
            else:
                minimumScore = float("inf")
                for action in gameState.getLegalActions(agentState):
                    # Find the minimum score of the ghost
                    score = alpha_beta_search(nextAgentState, nextDepth, gameState.getNextState(agentState, action), alpha, beta)
                    # Prune the branch
                    if score < alpha:
                        return score
                    # Update beta
                    beta = min(beta, score)
                    minimumScore = min(minimumScore, score)
                return minimumScore

        alpha = float("-inf")
        beta = float("inf")
        maximumScore = float("-inf")
        bestAction = gameState.getLegalActions(0)[0]

        # Iterate through all of Pacman's legal actions
        for action in gameState.getLegalActions(0):
            score = alpha_beta_search(1, 0, gameState.getNextState(0, action), alpha, beta)
            # Update the score and action when the score > maximum score
            if score > maximumScore:
                maximumScore = score
                bestAction = action
            # Update alpha
            alpha = max(alpha, maximumScore)

        return bestAction

        # End your code (Part 2)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)

        def terminalState(gameState, depth):
            """
            Check if the game is over or the depth limit is reached.

            Args:
                gameState: The current game state.
                depth: The current depth of the search tree.

            Returns:
                bool: True if the game is over or the depth limit is reached, False otherwise.
            """
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def expectimax_search(agent, depth, gameState):
            # Check if the game is over or the depth limit is reached
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            # Get the next agent and update depth when all agents have been traversed
            nextAgent = (agent + 1) % gameState.getNumAgents()
            nextDepth = depth + (1 if nextAgent == 0 else 0)

            if agent == 0:  # Pacman's turn
               
                bestScore = float("-inf")
                
                for action in gameState.getLegalActions(agent):
                    
                    score = expectimax_search(nextAgent, nextDepth, gameState.getNextState(agent, action))
                    # Update the best score
                    bestScore = max(bestScore, score)
                return bestScore
            else:  # Chance node (ghost)
               
                totalScore = 0
                # Get the legal actions for the current ghost
                legalActions = gameState.getLegalActions(agent)
                # Get the number of legal actions
                numActions = len(legalActions)
                
                for action in legalActions:
                    
                    totalScore += expectimax_search(nextAgent, nextDepth, gameState.getNextState(agent, action))
                # Calculate the expected score by averaging the total score over the number of actions
                return totalScore / float(numActions)

        bestScore = float("-inf")
        bestAction = None

        # Iterate through all of Pacman's legal actions
        for action in gameState.getLegalActions(0):
            score = expectimax_search(1, 0, gameState.getNextState(0, action))
            # Update the best score and action when a better score is found
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        # End your code (Part 3)



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """

    This function calculates the evaluation score for a given game state.
    It considers various factors such as Pacman's position, ghost positions, remaining food, and capsules to determine the score.
    The score is adjusted based on specific conditions within the game.

    Here's how it works:
        Pacman's score is initially set to 10 times the current score.
        If Pacman can eat a scared ghost, 30 points are added to the score for each unit of distance to the nearest scared ghost.
        If Pacman is within 3 units of a ghost, 10 points are added to the score for each unit of distance to the nearest ghost.
        Otherwise, the score is reduced by the minimum distance to the nearest food, the minimum distance to the nearest capsule, 5 times the number of remaining food pellets, and 100 times the number of remaining capsules.
   
     """
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = [(manhattanDistance(pacmanPos, currentGameState.getGhostPosition(Id)), Id) for Id in range(1, currentGameState.getNumAgents())]
    minGhostDist, _ = (0, 0) if len(ghostStates) == 0 else min(ghostStates)
    isEat = any(ghost.scaredTimer > 1 for ghost in currentGameState.getGhostStates())
    curScore = currentGameState.getScore()
    foodList = currentGameState.getFood().asList()
    numFood = len(foodList)
    minFoodDist = min([manhattanDistance(pacmanPos, food) for food in foodList], default=0)
    numCapsules = len(currentGameState.getCapsules())
    capsulesDist = [manhattanDistance(pacmanPos, capsule) for capsule in currentGameState.getCapsules()]
    minCapsuleDist = min(capsulesDist, default=0)

    # Evaluate based on conditions
    evaluation = 10 * curScore
    if isEat:
        evaluation += (-35 * minGhostDist)
    elif minGhostDist < 3:
        evaluation += (15 * minGhostDist)
    else:
        evaluation += (-1.5 * minFoodDist)
        evaluation += (-20 * minCapsuleDist)
        evaluation += (-5 * numFood)
        evaluation += (-100 * numCapsules)

    return evaluation
    # End your code (Part 4)

    
    






# Abbreviation
better = betterEvaluationFunction
