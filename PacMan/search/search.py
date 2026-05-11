# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
#from typing_extensions import override

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
        Search the deepest nodes in the search tree first.

        Your search algorithm needs to return a list of actions that reaches the
        goal. Make sure to implement a graph search algorithm.

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:

        print("Start:", problem.getStartState())
        print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        """
    "*** YOUR CODE HERE ***"
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    from util import Stack
    # incepem de la un nod de start
    # introducem intr-o lista nodul de start
    # introducem o lista drum pentru fiecare succesor al nodului curent
    # marcam nodurile vizitate
    # cat timp lista de succesori nu e goala:
        # luam primul element din ea
        # verificam isGoalState:
            # daca da -> return lista drum
        # verificam daca starea curenta nu a mai fost vizitata:
            # o marcam ca si vizitata
            # gasim toti succesorii:
                # ii adaugam in lista drumului nou
                # adaugam succesorul curent si lista drumului in lista
    list = Stack()
    list.push( (problem.getStartState(), []))
    visited = []

    while not list.isEmpty():
        currentState, currentPath = list.pop()

        if problem.isGoalState(currentState):
            return currentPath

        if currentState not in visited:
            visited.append(currentState)
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newPath = currentPath + [action]
                list.push((successor, newPath))
    return []

'''def breadthFirstSearch(problem):
    from util import Queue

    list = Queue()
    list.push((problem.getStartState(), []))
    visited = []

    while not list.isEmpty():
        currentState, currentPath = list.pop()

        if problem.isGoalState(currentState):
            print("PATH: ", currentPath)
            return currentPath

        if currentState not in visited:
            visited.append(currentState)
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newPath = currentPath + [action]
                print("successor path: ", [action])
                list.push((successor, newPath))
    return []'''


def breadthFirstSearch(problem):
    from util import Queue

    # 1. Rename 'list' to 'fringe' (good practice)
    queue = Queue()
    queue.push((problem.getStartState(), []))

    # 2. Use a SET, not a list.
    # This is 1000x faster for checking duplicates.
    visited = set()

    while not queue.isEmpty():
        currentState, currentPath = queue.pop()

        if problem.isGoalState(currentState):
            return currentPath

        # 3. Check against the set (Instant check)
        if currentState not in visited:
            # 4. Add to set (Use .add instead of .append)
            visited.add(currentState)

            for successor, action, stepCost in problem.getSuccessors(currentState):
                # Your path logic here is PERFECT. Keep it.
                newPath = currentPath + [action]
                queue.push((successor, newPath))

    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # incepem de la un nod de start
    # introducem intr-o lista nodul de start
    # introducem o lista pentru fiecare succesor al nodului curent
    # marcam nodurile vizitate
    # cat timp lista de succesori nu este goala:
        # luam primul element din lista
        # se verifica daca s-a ajuns la final
            # daca s-a ajuns la final -> return
        # verificam daca starea curenta nu a mai fost vizitata
            # o marcam ca si vizitata
            # gasim toti succesorii:
                # initializam o variabila care sa calculeze costul/drum
                # facem cate o lista noua pentru fiecare succesor
                # adaugam succesorul curent si lista cu drumul in lista

    list = util.PriorityQueue()
    list.push((problem.getStartState(), [], 0), 0)
    visited = []
    while not list.isEmpty():
        currentState, currentPath, currentCost = list.pop()

        if problem.isGoalState(currentState):
            #print(totalCost)
            return currentPath

        if currentState not in visited:
            visited.append(currentState)
            for successor, action, stepCost in problem.getSuccessors(currentState):
                newPath = currentPath + [action]
                newCost = currentCost + stepCost
                list.push((successor, newPath, newCost), newCost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # https://www.geeksforgeeks.org/dsa/a-search-algorithm/
    # gaseste cel mai scurt drum intre punctul a si b

    # pasii pe care ii face sunt in concordanta cu o functie f - care vecin are cea mai mica functie f
    # f = g + h; g - costul de la start pana la un anumit punct
    #            h - costul estimat de la punctul curent pana la final

    """ ALGORITM:
        - initializam lista deschisa
        - initializam lista inchisa
            - punem nodul de start in lista deschisa cu f=0
        - cat timp lista deschisa nu e goala
            - gasim nodul cu cel mai mic f si il numim q
            - pop la q din lista deschisa
            - generam cei 8 succesori ai q si le setam parintele ca q
            - pentru fiecare succesor
                - daca isGoalReached - return
                - daca nu, calculam g si h
                - daca gasim din nou un nod gasit anterior care se afla in lista inchisa si are f mai mic, dam skip la succesor
                  (adica daca pozitiile a 2 noduri coincid - un nod contine si costul si parintele, care sunt diferite)
                - daca nu, il adaugam la lista deschisa
        - push q in lista inchisa
    """
    from util import Stack, PriorityQueue
    from searchAgents import manhattanHeuristic
    openList = PriorityQueue()
    closedList = {}
    #((state, path, g), f)
    openList.push((problem.getStartState(), [], 0), 0)

    while not openList.isEmpty():
        currentState, currentPath, currentG = openList.pop()
        if problem.isGoalState(currentState):
            return currentPath
        if currentState in closedList and closedList[currentState] < currentG:
            continue
        closedList[currentState] = currentG
        for successor, action, stepCost in problem.getSuccessors(currentState):
            newG = currentG + stepCost
            newH = heuristic(successor, problem)
            newF = newG + newH
            newPath = currentPath + [action]
            openList.push((successor, newPath, newG), newF)
    return []
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
