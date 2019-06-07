#!/usr/bin/env python

import numpy as np 
import rospy 
import time 
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse
from robot_sim.srv import RobotPolicy
from robot_sim.srv import RobotPolicyRequest
from robot_sim.srv import RobotPolicyResponse

# network parameters
HINDDEN_SIZE = 300

# dqlearn parameters
CAPACITY = 10000
NUM_EPISODES = 800
NUM_STEPS = 500
LEARN_RATE = 0.01
DISCOUNT_RATE = 0.95
BATCH_SIZE = 64
TARGET_UPDATE = 100
SEED = 7
EPS_START = 0.9
EPS_END = 0.05
EPS_RATE = 500

class DQNet(nn.Module):
    def __init__(self):
        super(DQNet, self).__init__()
        self.fc1 = nn.Linear(4, HINDDEN_SIZE)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(HINDDEN_SIZE, 2)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        q_value = self.out(out)
        return q_value

class DQLearn(object):
    def __init__(self):
        self.act_net = DQNet()  # initialize action network
        self.t_net = DQNet()    # initialize target network
        self.memory = np.zeros((CAPACITY, 10))     # initialize memory
        self.optimizer = torch.optim.RMSprop(self.act_net.parameters(), lr = LEARN_RATE)
        self.memory_count = 0
        self.update_count = 0
        self.steps = 0
        
    def store(self, transition): 
        idx = self.memory_count % CAPACITY
        self.memory[idx,:] = transition
        self.memory_count += 1

    def select(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        q_value = self.act_net.forward(state)
        eps = EPS_END + (EPS_START-EPS_END)*math.exp(-1*self.steps/EPS_RATE)
        self.steps += 1
        q_max_idx = 0
        if np.random.uniform() > eps:
            q_max_idx = torch.max(q_value, 1)[1].data.numpy()[0]
        else:
            q_max_idx = np.random.randint(0,2)
        return q_max_idx

    def update(self):
        if self.update_count % TARGET_UPDATE == 0:
            self.t_net.load_state_dict(self.act_net.state_dict())
        self.update_count += 1

    def learn(self, useful_memory):
        self.update()
        sample_idx = np.random.choice(np.shape(useful_memory)[0], BATCH_SIZE)
        minibatch = useful_memory[sample_idx,:]
        state = torch.FloatTensor(minibatch[:,:4])
        action = torch.LongTensor(minibatch[:,4]).unsqueeze(1)
        reward = torch.FloatTensor(minibatch[:,5]).unsqueeze(1)
        state_new = torch.FloatTensor(minibatch[:,6:])
        q_act_value = self.act_net.forward(state).gather(1, action)
        q_t_value = self.t_net.forward(state_new).detach()
        # find terminal and replace q-value to 0
        if np.isnan(minibatch[:,6]).any():
            nan_idx = np.argwhere(np.isnan(minibatch[:,6]))
            for i in nan_idx:
                q_t_value[i[0],0] = 0
                q_t_value[i[0],1] = 0
        q_expect_value = reward + DISCOUNT_RATE*q_t_value.max(1)[0].view(-1,1)
        loss = F.smooth_l1_loss(q_act_value, q_expect_value)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.act_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()

    def train(self):
        self.episode_durations = []
        env = Environment()
        self.episode = 0
        t_start = time.time()
        for episode in range(NUM_EPISODES):
            t_now = time.time()
            if (t_now - t_start)>280:
                break
            self.reward = 0
            self.episode += 1
            print ("epi:",episode)
            state = env.get_start_state()
            for step in range(NUM_STEPS):
                action_idx = self.select(state)
                state_new = env.get_state_reward(action_idx)[0]
                reward = env.get_state_reward(action_idx)[1]
                self.reward +=1
                transition = np.zeros((1,10))[0]
                transition[:4] = state
                transition[4] = action_idx
                transition[5] = reward
                transition[6:] = state_new
                self.store(transition)
                if self.memory_count >= BATCH_SIZE:
                    if self.memory_count < CAPACITY:
                        useful_memory = self.memory[:self.memory_count,:]
                    else:
                        useful_memory = self.memory
                    self.learn(useful_memory)
                state = state_new
                if np.isnan(state).any():
                    print step
                    break

    def test(self):
        self.server = rospy.Service('cartpole_policy', RobotPolicy, self.send_action)
        print "Service begin ..."
        rospy.spin()
        
    def send_action(self, req):
        state = req.robot_state
        state = torch.tensor(state, dtype = torch.float).unsqueeze(0)
        q_value = self.act_net.forward(state)
        q_max_idx = torch.max(q_value, 1)[1].data.numpy()[0]
        env = Environment()
        action = env.action_list[q_max_idx]
        return RobotPolicyResponse([action])
        
class Environment(object):
    def __init__(self):
        self.action_list = [-10.0,10.0]
        self.robot_state = rospy.ServiceProxy('cartpole_robot', RobotAction)
        self.x_max = 1.2
        self.angle_max = 6*np.pi/180

    def get_start_state(self):
        angle = np.random.randint(-3,4)
        angle = angle*np.pi/180
        req = RobotActionRequest()
        req.reset_robot = True
        req.reset_pole_angle = angle
        resp = self.robot_state(req)
        return resp.robot_state

    def get_state_reward(self, action_idx):
        action = self.action_list[action_idx]
        req = RobotActionRequest()
        req.reset_robot = False
        req.action = [action]  # type is list
        resp = self.robot_state(req)
        state_new = resp.robot_state
        reward = 1
        if (abs(state_new[0]) > self.x_max) | (abs(state_new[1]) > self.angle_max) :
            state_new = [np.nan, np.nan, np.nan, np.nan]
            reward = 0
        return [state_new, reward]

if __name__ == '__main__':
    rospy.init_node('cartpole_policy', anonymous=True)
    np.random.seed(SEED)
    dql = DQLearn()
    dql.train()
    print "Successfully train the network!"
    dql.test()
    print "Over!"
