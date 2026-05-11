# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Implements the batch version of Value Iteration for 'self.iterations' rounds.

        # V_k is stored in self.values.
        # We will use a temporary Counter (V_k_plus_1) for the new values.

        for i in range(self.iterations):
            # We must use the values from the current iteration (self.values)
            # to compute ALL values for the next iteration (V_k_plus_1).
            V_k = self.values.copy()  # V_k stores values from iteration k
            V_k_plus_1 = util.Counter()  # V_k_plus_1 will store values for iteration k+1

            # Iterate over all states in the MDP
            for state in self.mdp.getStates():

                # Terminal states have a value of 0 and no actions.
                if self.mdp.isTerminal(state):
                    V_k_plus_1[state] = 0
                    continue

                # Find the maximum Q-value for the state using V_k.
                max_q_value = -float('inf')

                # Iterate over all possible actions
                for action in self.mdp.getPossibleActions(state):
                    # Compute Q(s, a) using V_k.
                    # Note: We must call computeQValueFromValues, but pass V_k explicitly
                    # since computeQValueFromValues uses self.values, which is V_k_plus_1
                    # mid-update. The prompt implies computeQValueFromValues should use self.values,
                    # so we will temporarily set self.values to V_k for this computation
                    # for states where we are *not* the current state.

                    # A cleaner approach is to implement the Q-value calculation inline
                    # for runValueIteration to ensure correct batch update logic.

                    q_value = 0
                    # Iterate over successor states and probabilities
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        reward = self.mdp.getReward(state, action, nextState)
                        # The successor value V_k(s') MUST come from the fixed V_k (stored in V_k).
                        nextStateValue = V_k[nextState]

                        q_value += prob * (reward + self.discount * nextStateValue)

                    # Update the max Q-value for the state
                    max_q_value = max(max_q_value, q_value)

                # Set the new value for the state V_k+1(s)
                V_k_plus_1[state] = max_q_value

            # After computing ALL V_k+1 values, update self.values for the next iteration
            self.values = V_k_plus_1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return 0.0

        q_value = 0
        # Iterate over successor states and probabilities
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)

            # The value of the successor state V(s') comes from the current self.values
            nextStateValue = self.getValue(nextState)

            q_value += prob * (reward + self.discount * nextStateValue)

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)
        if not possibleActions:
            return None

        bestAction = None
        maxQValue = -float('inf')

        # Find the action that maximizes the Q-value
        for action in possibleActions:
            # Use the already implemented helper to get the Q-value
            q_value = self.computeQValueFromValues(state, action)

            if q_value > maxQValue:
                maxQValue = q_value
                bestAction = action

        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        # 1. Compute predecessors of all states
        # Predecessors[s'] will store a set of predecessor states {s} that can reach s'
        predecessors = {}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                            # Store ONLY the predecessor state, as required by the unique iteration in step 4.d
                        predecessors[nextState].add(state)

                        # 2. Initialize an empty priority queue
        priority_queue = util.PriorityQueue()

        # 3. For each non-terminal state s, do:
        # Must iterate over self.mdp.getStates() for the autograder (as specified in the notes).
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):

                # a. Calculate the highest Q-value (Q_max) for s (the *should-be* value)
                q_values = [self.computeQValueFromValues(s, action)
                            for action in self.mdp.getPossibleActions(s)]

                if not q_values:  # Handle non-terminal state with no actions
                    continue

                q_max = max(q_values)

                # Calculate the difference (error)
                diff = abs(self.getValue(s) - q_max)  # self.values[s] is used via getValue(s)

                # b. If diff > theta, push s into the priority queue with priority -diff.
                if diff > self.theta:
                    priority_queue.push(s, -diff)

        # 4. For iteration 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):

            # a. If the priority queue is empty, then terminate.
            if priority_queue.isEmpty():
                break

            # b. Pop a state s off the priority queue.
            s = priority_queue.pop()

            # c. Update the value of s (online update using current self.values)
            if not self.mdp.isTerminal(s):
                q_values = [self.computeQValueFromValues(s, action)
                            for action in self.mdp.getPossibleActions(s)]

                if q_values:
                    # Update V(s) to the new value (max Q-value)
                    self.values[s] = max(q_values)

            # d. For each predecessor p of s, do:
            # We iterate over the unique predecessor states (p) directly
            for p in predecessors.get(s, set()):
                if not self.mdp.isTerminal(p):

                    # i. Find the highest Q-value across all possible actions from p (Q_max_p).
                    q_values_p = [self.computeQValueFromValues(p, action)
                                  for action in self.mdp.getPossibleActions(p)]

                    if not q_values_p: continue

                    q_max_p = max(q_values_p)

                    # Calculate the difference (error)
                    diff = abs(self.getValue(p) - q_max_p)

                    # ii. If diff > theta, push/update p into the priority queue with priority -diff.
                    if diff > self.theta:
                        # The update method handles existence and priority update automatically.
                        priority_queue.update(p, -diff)

