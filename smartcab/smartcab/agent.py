import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.state = None
        self.qTable = {}
        self.gamma = 0.2 # discount rate
        self.alpha = 0.8 # learning rate
        self.epsilon = 0.9
        self.totalPenalties = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required
        self.totalPenalties = 0
        self.epsilon = self.epsilon/1.2

    def setQ(self, state, action, value):
        self.qTable[(state, action)] = value

    def getQ(self, state, action):
        return self.qTable.get((state, action), 0.0)

    def getCurrentState(self):
        inputs = self.env.sense(self)
        inputs['waypoint'] = self.planner.next_waypoint()

        del inputs['oncoming']
        del inputs['left']
        del inputs['right']

        return tuple(inputs.items())

    def getBestAction(self):
        # Basic agent
        # action = random.choice(self.actions)

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

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
