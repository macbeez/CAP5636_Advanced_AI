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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        V_values = []
        from matplotlib import pyplot as plt
        for i in range(self.iterations):
            updated_value = self.values.copy()
            mdp_states = self.mdp.getStates()
            for state in mdp_states:
                # print("Current state: ", state)
                maxVal = float('-inf')
                if self.mdp.isTerminal(state):
                    continue
                else:
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        value = self.computeQValueFromValues(state, action)
                        # print("Action and value from state :", action, value)
                        if value > maxVal:
                            maxVal = value
                    updated_value[state] = maxVal
                    # print("updated_value[state]: ", updated_value[state])
                if state == (0,2):
                    V_values.append(updated_value[state])
                    print("The values of the state (0,2) are: ", V_values)
            self.values = updated_value
            # print("States and the corresponding values: ", self.values)
        plt.plot(V_values)
        # plt.xlim(0, 20) 
        plt.xlabel("Iterations")
        plt.ylabel("V values")
        plt.title("Evolution of V values for state (0,2)")
        plt.show()
            

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
        Q_value = 0
        states_and_probs = self.mdp.getTransitionStatesAndProbs(state,action)
        # print("Transition states and probabilities: ", states_and_probs)
        for trans_state_and_prob in states_and_probs:
            transition_state = trans_state_and_prob[0]
            transition_prob = trans_state_and_prob[1]
            # print("Transition state: ", transition_state)
            # print("Transition probability: ", transition_prob)
            reward = self.mdp.getReward(state, action, transition_state)
            # print("Reward obtained: ", reward)
            # print("\n")
            ## Q - Value formula: summation over s'[T(s,a,s'){R(s,a,s') + discount * V(s')}]
            Q_value += transition_prob * (reward + self.discount * self.getValue(transition_state))
            # print("Q value for the state", transition_state, " and action ", action, "is: ", Q_value)
        return Q_value
        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxVal = float('-inf')
        state_action = "north"
        if not self.mdp.isTerminal(state):
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                value = self.computeQValueFromValues(state, action)
                # print("Action and value from state :", action, value)
                if value > maxVal:
                    maxVal = value
                    state_action = action
        return state_action

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
