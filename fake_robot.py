#!/usr/bin/env python

import numpy as np 
import rospy 
import time 
from sklearn.linear_model import LinearRegression

from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse

class FakeRobot(object):
    def __init__(self, num_actions, num_steps):
        self.num_actions = num_actions
        self.num_steps = num_steps
        self.robot_new_state = rospy.ServiceProxy('real_robot', RobotAction)
        self.train_data_matrix = np.zeros((self.num_actions,self.num_steps,15)) # (action, step, (feature, prediction))

    def get_train_data(self):
        for k in range(0,self.num_actions):
            self.train_data_matrix[k,0,0:6] = [-1.57,0.0,0.0,0.0,0.0,0.0]
            action = np.random.rand(1,3)
            action[0,0] = (2 * action[0,0] - 1.0) * 1.0
            action[0,1] = (2 * action[0,1] - 1.0) * 0.5
            action[0,2] = (2 * action[0,2] - 1.0) * 0.25
            self.get_new_state(action, k)
            time.sleep(1.00)

    def get_new_state(self, action, k):
        req = RobotActionRequest()
        req.reset = True
        resp = self.robot_new_state(req)
        for i in range(self.num_steps):
            req = RobotActionRequest()
            req.reset = False
            req.action = action.reshape((3))
            resp = self.robot_new_state(req)
            self.train_data_matrix[k,i,6:9] = action.reshape((3))
            self.train_data_matrix[k,i,9:15] = resp.robot_state
            time.sleep(0.01)
        self.train_data_matrix[k,1:self.num_steps,0:6] = self.train_data_matrix[k,0:self.num_steps-1,9:15]

    def get_model(self):
        train_matrix = self.train_data_matrix.reshape((self.num_actions*self.num_steps,15))
        self.model = LinearRegression().fit(train_matrix[:,0:9],train_matrix[:,9:15])

    def server(self):
        self.server = rospy.Service('fake_robot', RobotAction, self.predict_state)
        print "Service begin ..."
        rospy.spin()
      
    def predict_state(self, req):
        if req.reset == True:
            self.state = np.zeros((1,15))[0]
            init_state = [-1.57,0.0,0.0,0.0,0.0,0.0]
            self.state[0:6] = init_state
        else:
            self.state[6:9] = req.action
            self.state[9:15] = self.model.predict(np.reshape(self.state[0:9],(1,-1)))
            self.state[0:6] = self.state[9:15]
        return RobotActionResponse(self.state[9:15])

if __name__ == '__main__':
    rospy.init_node('fake_robot', anonymous=True)
    fakerobot = FakeRobot(10,20)
    fakerobot.get_train_data()
    print "Successfully get train data!"
    fakerobot.get_model()
    print "Successfully get model!"
    fakerobot.server()
    print "Over!"