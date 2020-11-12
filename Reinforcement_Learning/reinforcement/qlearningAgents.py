# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import csv

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.q_val = util.Counter() # dictionary with values for key = 0
        self.action_north = []
        self.action_south = []
        self.action_east = []
        self.action_west = []
        self.q_actions = {"north": [], "south": [], "east": [], "west": []}
        self.episode = 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # The counter class returns value 0 for a state it hasn't seen
        return self.q_val[(state,action)]
        
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        maxVal = float('-inf')
        actions = self.getLegalActions(state)
        # print("Actions: ", actions)
        if not len(actions):
          return 0.0
        else:
          for action in actions:
            q_value = self.getQValue(state, action)
            # print("Q-value: ", q_value)
            if q_value > maxVal:
              maxVal = q_value
          return maxVal

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxVal = float('-inf') # set default values
        maxAct = "north" 
        actions = self.getLegalActions(state)
        for action in actions:
          q_value = self.getQValue(state, action)
          if q_value > maxVal:
            maxVal = q_value
            maxAct = action
        return maxAct

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        random_action = util.flipCoin(self.epsilon)
        if random_action:
          # pick a random action from the set of actions. 
          action = random.choice(legalActions)
          # print("The randomly picked action is: ", action)
        else:
          action = self.computeActionFromQValues(state)

        return action

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q_value formula: 
        # (1-alpha)*current_Qvalue + alpha*[reward + discount*Qvalue(next_state)]

        current_Qvalue = self.getQValue(state, action)
        next_state_value = self.computeValueFromQValues(nextState)
        updated_Qvalue = (1-self.alpha)*current_Qvalue + self.alpha * (reward + self.discount * next_state_value)
        self.q_val[(state,action)] = updated_Qvalue
        # print("final answer to the update: ", self.q_val[(state,action)])

        if state == (3,2) or state == (3,1):
          self.q_actions['north'].append(self.q_val[((1,2),'north')])
          self.q_actions['south'].append(self.q_val[((1,2),'south')])
          self.q_actions['east'].append(self.q_val[((1,2),'east')])
          self.q_actions['west'].append(self.q_val[((1,2), 'west')])

          # creating a 2D list for excel data dump
        csv_output = [["episode", "north", "south", "east", "west"]] # first row of the csv
        for row in range(len(self.q_actions['east'])):
          # append value for each action based on the current iteration (row)
          csv_output.append([row+1, self.q_actions['north'][row], self.q_actions['south'][row], self.q_actions['east'][row], self.q_actions['west'][row]])
        with open("step3_output_1000.csv","w+") as my_csv:
          csvWriter = csv.writer(my_csv,delimiter=',')
          csvWriter.writerows(csv_output)

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
