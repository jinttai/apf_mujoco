import numpy as np
from stl import mesh
import open3d as o3d
import spart_functions as spart
import apf_functions as apf

def get_closest_point_link(rJ, Rj, link_length, pcd):
    # rJ: position of the joint / inertial frame
    # Rj: orientation of the joint / inertial frame
    # pcd: point cloud data / inertial frame
    pcd_joint = np.dot(Rj.T, pcd-rJ)
    candidate_point_joint = pcd_joint[(pcd_joint[:, 2] >= 0) & (pcd_joint[:, 2] <= link_length)]
    if candidate_point_joint.size == 0:
        point_min = np.array([0.0, 0.0, 0.0])
        dis_min = - 1.0
        return point_min, dis_min
    else:
        idx = np.argmin(np.linalg.norm(candidate_point_joint[:, :2], axis=1))
        point_min_joint = candidate_point_joint[idx]
        dis_min = np.linalg.norm(candidate_point_joint[idx,:2])
        return point_min_joint, dis_min

def get_generalized_jacobian(J_0k, J_mk, H0, H0m):
    gjm = -np.dot(np.dot(J_0k, np.linalg.pinv(H0)), H0m) + J_mk
    return gjm

def repulsive_potential_field_link(x, pcd, r_min, rJ, Rj, link_length):
    point_min_joint, dis_min = get_closest_point_link(rJ, Rj, link_length, pcd)
    k_r = 0.5
    if dis_min < r_min and dis_min > 0.0:
        return k_r * 0.5 * (1 / dis_min) ** 2
    else:
        return 0
    
def repulsive_force_link_joint(pcd, r_min, rJ, Rj, link_length):
    point_min_joint, dis_min = get_closest_point_link(rJ, Rj, link_length, pcd)
    k_r = 0.5
    if dis_min < r_min and dis_min > 0.0:
        force_direction_joint = - [point_min_joint[0], point_min_joint[1], 0.0] 
        return -k_r * (- force_direction_joint) / dis_min ** 3
    else:
        return np.array([0.0, 0.0, 0.0])

def attractive_force_link(goal, d_goal,r_hug, rJ, Rj, x_1, x_td, link_length):
    #r_hug :radius of hugging circle
    #x_1 : position of first joint
    #x_td : direction of the target
    k_a1 = 0.5
    k_a2 = 0.5
    x = np.dot(Rj.T, link_length/2) + rJ
    d = np.linalg.norm(x - goal)
    if abs(d - r_hug) < d_goal:
        return k_a1 * (d - r_hug) * (x - goal) / d + k_a2 * x_td * np.sign(np.dot(x - x_1, x_td))
    else:
        return k_a1 * (x - goal) / np.linalg.norm(x - goal)


def force_to_torque(F, J_0k, J_mk, H0, H0m):
    gjm_k = get_generalized_jacobian(J_0k, J_mk, H0, H0m)
    tau = np.dot(gjm_k.T, F)
    return tau

def total_torque(robot, r0, rL, P0, pm, rJ, Rj, H0, H0m, link_length, pcd, goal, d_goal, r_min):
    # goal : goal position for hugging
    # d_goal : threshold distance 
    n = robot['n_links_joints']
    F_r = np.zeros((3, n))
    F_a = np.zeros((3, n))
    torque = np.zeros((n, 1))
    for i in range(n):
        F_r_joint = repulsive_force_link_joint(pcd, r_min, rJ[:,i], Rj[:,:,i], link_length[i])
        F_r[:,i] = np.dot(Rj[:,:,i].T, F_r_joint)
        F_a[:,i] = attractive_force_link(goal, d_goal, rJ[:,i], Rj[:,:,i], link_length[i])
        F_k = F_a[:,i] + F_r[:,i]
        rp = rJ[:,i]
        J_0k, J_mk = spart.jacobians(rp, r0, rL, P0, pm, i, robot)
        torque += force_to_torque(F_k, J_0k, J_mk, H0, H0m)
    return torque




    