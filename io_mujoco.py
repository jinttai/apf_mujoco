#!/usr/bin/env python3

import os 
import rospkg
import rospy
import mujoco
import mujoco.viewer
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

    
def control(model,data,control):
    data.ctrl = control

def get_states(model,data):
    qpos = data.qpos
    qvel = data.qvel
    return qpos, qvel
    
def pub_states(model,data):
    n = model.nv
    H = np.zeros((n,n))
    C = np.zeros((n))
    J = np.zeros((6,n))
    p_ee = np.zeros((3,1))

    msg_s = JointState()
    msg_s.position = data.qpos.tolist()
    msg_s.velocity = data.qvel.tolist()    

    msg_m = Float64MultiArray()
    msg_m.layout.dim.append(MultiArrayDimension())
    msg_m.layout.dim.append(MultiArrayDimension()) # 2D array 
    msg_m.layout.dim[0].label = "rows"
    msg_m.layout.dim[0].size = n
    msg_m.layout.dim[0].stride = n*n
    msg_m.layout.dim[1].label = "cols"
    msg_m.layout.dim[1].size = n 
    msg_m.layout.dim[1].stride = n 

    msg_m.data = H.flatten().tolist() 

    msg_c = Float64MultiArray()
    msg_c.layout.dim.append(MultiArrayDimension())

    msg_c.layout.dim[0].label = "rows"   
    msg_c.layout.dim[0].size = n
    msg_c.layout.dim[0].stride = n 
    

    msg_c.data = C 

    msg_j = Float64MultiArray()
    msg_j.layout.dim.append(MultiArrayDimension())
    msg_j.layout.dim.append(MultiArrayDimension())
    msg_j.layout.dim[0].label = "rows"  
    msg_j.layout.dim[0].size = 6
    msg_j.layout.dim[0].stride = 6*n
    msg_j.layout.dim[1].label = "cols"      
    msg_j.layout.dim[1].size = n 
    msg_j.layout.dim[1].stride = n 
    msg_j.data = J.flatten().tolist()
    

    msg_ee = Float64MultiArray() 
    msg_ee.layout.dim.append(MultiArrayDimension())
    msg_ee.layout.dim[0].label ="rows"
    msg_ee.layout.dim[0].size = 3 
    msg_ee.layout.dim[0].stride = 3 
    msg_ee.data = p_ee # end-effector here 


    # Publish M & C matrices
    state_pub.publish(msg_s)
    M_pub.publish(msg_m)    
    C_pub.publish(msg_c)
    J_pub.publish(msg_j)
    xee_pub.publish(msg_ee)
    #rospy.loginfo(f"end effector at: {msg_ee.data}")
    
q_size = 10
# Publishers
state_pub = rospy.Publisher("/mujoco/state", JointState, queue_size=q_size)
M_pub = rospy.Publisher("/mujoco/M", Float64MultiArray, queue_size=q_size)
C_pub = rospy.Publisher("/mujoco/C", Float64MultiArray, queue_size=q_size)
J_pub = rospy.Publisher("/jacobian", Float64MultiArray,queue_size=q_size)
xee_pub = rospy.Publisher("/r_ee", Float64MultiArray,queue_size=q_size)
