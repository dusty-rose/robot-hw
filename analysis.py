#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hand_analysis.msg import GraspInfo

class Analysis(object):
    def __init__(self):
    	file = rospy.get_param('~train_filename')
        train_data = np.genfromtxt(fname = file, delimiter = ',', skip_header = 1)
        train_label = train_data[:,0]
        train_emg = train_data[:,1:9]
        train_glove = train_data[:,9:24]
        self.GtoL = KNeighborsClassifier().fit(train_glove, train_label)
        self.scaler = StandardScaler().fit(train_emg) 
        standard_train_emg = self.scaler.transform(train_emg)
        self.EtoL = SVC().fit(standard_train_emg, train_label)
        self.EtoG = LinearRegression().fit(train_emg, train_glove)
        self.DimReduc = PCA(n_components = 2).fit(train_glove)
        self.sub = rospy.Subscriber("/grasp_info", GraspInfo, self.callback, queue_size = 100)
        self.pub = rospy.Publisher("/labeled_grasp_info", GraspInfo, queue_size = 100)
    
    def callback(self, msg):
        if msg.label == -1:
            if msg.glove:
                self.test_glove = np.reshape(np.array(msg.glove),(1,-1))
                self.predict_label = self.GtoL.predict(self.test_glove) 
                msg.label = self.predict_label
                self.pub.publish(msg)
            else:
                self.test_emg = np.reshape(np.array(msg.emg),(1,-1))
                self.standard_test_emg = self.scaler.transform(self.test_emg)
                self.predict_label = int(self.EtoL.predict(self.standard_test_emg)[0])
                msg.label = self.predict_label
                self.predict_glove = self.EtoG.predict(self.test_emg)
                msg.glove = list(self.predict_glove[0])
                self.pub.publish(msg)
        
        if msg.glove_low_dim:
            e = msg.glove_low_dim
            self.predict_glove = self.DimReduc.inverse_transform(e)
            msg.glove = self.predict_glove
            self.pub.publish(msg)
    
if __name__ == '__main__':
    rospy.init_node('analysis', anonymous = True)
    a = Analysis()
    rospy.spin()




