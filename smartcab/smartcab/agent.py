import random
import pandas as pd
import numpy as np

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    # Gamma: Discount rate
    # Alpha: Learning rate
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.actions = [None, 'forward', 'left', 'right']

        # TODO: Initialize any additional variables here
        self.resetAttributes()

    def resetAttributes(self, gamma=0.2, alpha=0.8, epsilon=0.9):
        self.state = None
        self.qTable = {}

        self.destinationReached = False
        self.totalRewards = 0
        self.totalPenalties = 0
        self.totalActions = 0

        self.penaltyFreeCount = 0
        self.destinationReachedCount = 0

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def reset(self, destination=None):
        self.planner.route_to(destination)

        if self.totalPenalties == 0:
            self.penaltyFreeCount += 1

        if self.destinationReached:
            self.destinationReachedCount += 1

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.destinationReached = False
        self.totalPenalties = 0
        self.epsilon = self.epsilon/1.2 # Decaying epsilon such that it favors exploration more at the start.

    def setQ(self, state, action, value):
        self.qTable[(state, action)] = value

    def getQ(self, state, action):
        return self.qTable.get((state, action), 0.0)

    def getCurrentState(self):
        inputs = self.env.sense(self)
        inputs['waypoint'] = self.planner.next_waypoint()

        # del inputs['left']
        # del inputs['right']
        # del inputs['oncoming']
        return tuple(inputs.items())

    def getBestAction(self):
        # For basic agent: -random choice-
        # action = random.choice(self.actions)

        # Sometimes, choose to explore instead of exploit
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            qValues = [self.getQ(self.state, a) for a in self.actions]
            maxQ = max(qValues)

            # If there is a tie
            count = qValues.count(maxQ)
            if count > 1:
                i = random.choice([i for i in range(len(self.actions)) if qValues[i] == maxQ])
            else:
                i = qValues.index(maxQ)

            action = self.actions[i]

        return action

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = self.getCurrentState()

        # TODO: Select action according to your policy
        action = self.getBestAction()

        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward < 0:
            self.totalPenalties += reward
        else:
            self.totalRewards += reward

        self.totalActions += 1

        # TODO: Learn policy based on state, action, reward
        s2 = self.getCurrentState()
        oldVal = self.getQ(self.state, action)
        maxQ2 = max([self.getQ(s2, a) for a in self.actions])
        newVal = oldVal + self.alpha * (reward + self.gamma * maxQ2 - oldVal)
        self.setQ(self.state, action, newVal)

        # print 'Proposed: ', self.next_waypoint
        # print 'Action: ', action
        # print 'Max Q2:', maxQ2
        # print 'Old Q Value: ', oldVal
        # print 'New Q Value: ', newVal
        # print 'Total penalties: ', self.totalPenalties
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    #### Values used for testing ####
    # gammas = [0.1, 0.2, 0.3, 0.4, 0.5] # discount
    # alphas = [0.5, 0.6, 0.7, 0.8, 0.9] # learning

    gammas = [0.1]
    alphas = [0.8]
    epsilon = 0.9

    df = pd.DataFrame(columns=['Gamma', 'Alpha', 'PenaltyFree', 'DestinationReached', 'Reward/Action Ratio'])

    index = 0

    # create environment (also adds some dummy traffic)
    e = Environment()
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    for gamma in gammas:
        for alpha in alphas:
            penaltyFreeCounts = []
            destinationReachedCounts = []
            rewardActionValues = []

            # Now simulate it 3 times for average of values
            for z in range(3):
                # This resets qTable as well so that multiple simulations of 100 runs are not artificially enhanced
                a.resetAttributes(alpha=alpha, gamma=gamma, epsilon=epsilon)
                sim.run(n_trials=100)  # press Esc or close pygame window to quit

                penaltyFreeCounts.append(a.penaltyFreeCount)
                destinationReachedCounts.append(a.destinationReachedCount)
                rewardActionValues.append(float(a.totalRewards)/a.totalActions)

            df.loc[index] = [
                    gamma,
                    alpha,
                    np.mean(penaltyFreeCounts),
                    np.mean(destinationReachedCounts),
                    np.mean(rewardActionValues)
                    ]

            index += 1

    print df

if __name__ == '__main__':
    run()
