from sls.learning.DQN_Learning import DQN_Learning
from sls.agents import AbstractAgent
from sls.minigames.utils import state_of_marine

import numpy as np

class QAgent(AbstractAgent):

    def __init__(self, train, screen_size):
        super(QAgent, self).__init__(screen_size)

        self.learner = DQN_Learning(range(8),
                                  epsilon=1,
                                  descending_epsilon=True,
                                  descend_epsilon_until=500,
                                  alpha=0.2)

    def step(self, obs):

        if obs.first():
            self.learner.descend_epsilon()
            return self._SELECT_ARMY

        beacon = self.get_beacon(obs)
        marine = self.get_marine(obs)
        state = state_of_marine(marine, beacon, self.screen_size, 10)
        #print("st: "+str(state))
        done = state == (0, 0)
        action, previous_state = self.learner.get_action(state, obs.reward)
        self.learner.remember(np.asarray(previous_state), action, obs.reward, np.asarray(state), done)
        self.learner.replay()
        self.learner.target_train()

        dest = []

        if action == 0:
            dest = [0, self.screen_size]
        elif action == 1:
            dest = [self.screen_size, 0]
        elif action == 2:
            dest = [0, -self.screen_size]
        elif action == 3:
            dest = [-self.screen_size, 0]
        elif action == 4:
            dest = [self.screen_size, self.screen_size]
        elif action == 5:
            dest = [self.screen_size, -self.screen_size]
        elif action == 6:
            dest = [-self.screen_size, self.screen_size]
        elif action == 7:
            dest = [-self.screen_size, -self.screen_size]

        return self._MOVE_SCREEN("now", self._xy_offset(self.get_unit_pos(marine), dest[0], dest[1]))

    def save_model(self, filename):
        self.learner.save_q_table(filename + '/model.pkl')

    def load_model(self, filename):
        self.learner.load_model(filename + '/model.pkl')

    def create_model(self):
        self.learner.create_models()
