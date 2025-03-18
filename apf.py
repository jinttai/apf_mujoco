#!/usr/bin/env python3
import numpy as np
import os
import rospy
import mujoco
import mujoco.viewer
import spart_python_code.spart_class as spart
import spart_python_code.urdf2robot as urdf2robot
import functions_link as fl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

import io_mujoco as io

def quat_dcm(q):
    """
    Convert a quaternion (q0, q1, q2, q3) to a 3x3 direction cosine matrix.
    """
    q0, q1, q2, q3 = np.asarray(q).flatten()
    return np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3),     2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3),     1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2),     2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)]
    ])

def apf_control_value(qpos, qvel, robot, rkd):
    
    r0 = qpos[0:3]
    quat0 = qpos[3:7]
    qm = qpos[7:12]
    R0 = quat_dcm(quat0)
    u0 = qvel[0:6]
    um = qvel[6:12]
    rkd.update_state(R0, r0, qm, um, u0)
    
    link_length = np.array([0.176, 0.613, 0.571, 0.135, 0.12, 0.18])
    goal = np.array([0.5, 0.5, 0.5])
    d_goal = 0.1
    r_min = 0.1
    
    control = fl.total_torque(robot, rkd.r0, rkd.rL, rkd.P0, rkd.pm, rkd.rJ, rkd.Rj, rkd.H0, rkd.H0m, link_length, pcd, goal, d_goal, r_min)
    
    return control

def main():
    rospy.init_node('apf')
    rate = rospy.Rate(10)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # initialize the model and data for mujoco
    model_path = os.path.join(current_dir, 'model_cjt', 'spacerobot_cjt.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # initialize the model for spart
    urdf_path = os.path.join(current_dir, 'model_cjt', 'spacerobot_cjt.urdf')
    robot_spart, robot_key = urdf2robot.urdf2robot(urdf_path)
    rkd = spart.RobotKinematicsDynamics(robot_spart)
    
    # parameters for the APF
    goal = np.array([0.5, 0.5, 0.5])
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while True:
            mujoco.mj_step(model, data)
            qpos, qvel = io.get_states(model, data)
            control = apf_control_value(qpos, qvel, robot_spart, rkd)
            io.control(model, data, control)
            viewer.sync()
            rate.sleep()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    