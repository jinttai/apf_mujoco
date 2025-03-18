#!/usr/bin/env python3
import numpy as np
import os
import rospy
import mujoco
import mujoco.viewer
from stl import mesh
import OpenGL.GL as gl
import spart_python_code.spart_class as spart
import spart_python_code.urdf2robot as urdf2robot
# import functions_link as fl
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

import io_mujoco as io

def load_stl_file(filename):
    # STL 파일 로드
    stl_mesh = mesh.Mesh.from_file(filename)
    # stl_mesh.vectors 의 shape은 (n_triangles, 3, 3) 입니다.
    points = stl_mesh.vectors.reshape(-1, 3)  # 모든 꼭짓점을 하나의 배열로 평면화
    # 중복되는 점 제거
    points = np.unique(points, axis=0)
    return points

def add_point_cloud_to_model(model_path, points, radius=0.01, rgba=[1, 0, 0, 1]):
    """
    Adds point cloud as small spheres to the Mujoco model and returns a new model
    
    Args:
        model_path: Path to the original model XML
        points: Nx3 array of point cloud coordinates
        radius: Radius of the spheres representing points
        rgba: Color and transparency of the points [red, green, blue, alpha]
    
    Returns:
        Modified Mujoco model with point cloud
    """
    # Read the model XML file
    with open(model_path, 'r') as f:
        xml_content = f.read()
    
    # Find the closing worldbody tag to insert our point cloud
    insert_pos = xml_content.find('</worldbody>')
    if insert_pos == -1:
        raise ValueError("Could not find </worldbody> tag in the XML")
    
    # Create point cloud spheres XML
    point_cloud_xml = ""
    for i, point in enumerate(points):
        point_xml = f"""
        <body name="point_{i}" pos="{point[0]} {point[1]} {point[2]}" mocap="true">
            <geom name="pointgeom_{i}" type="sphere" size="{radius}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0"/>
        </body>
        """
        point_cloud_xml += point_xml
    
    # Insert point cloud XML into the model
    new_xml_content = xml_content[:insert_pos] + point_cloud_xml + xml_content[insert_pos:]
    
    # Create a temporary XML file with the modified content
    temp_model_path = model_path.replace('.xml', '_with_points.xml')
    with open(temp_model_path, 'w') as f:
        f.write(new_xml_content)
    
    # Load the modified model
    model = mujoco.MjModel.from_xml_path(temp_model_path)
    return model, temp_model_path



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

# def apf_control_value(qpos, qvel, robot, rkd):
    
#     r0 = qpos[0:3]
#     quat0 = qpos[3:7]
#     qm = qpos[7:12]
#     R0 = quat_dcm(quat0)
#     u0 = qvel[0:6]
#     um = qvel[6:12]
#     rkd.update_state(R0, r0, qm, um, u0)
    
#     link_length = np.array([0.176, 0.613, 0.571, 0.135, 0.12, 0.18])
#     goal = np.array([0.5, 0.5, 0.5])
#     d_goal = 0.1
#     r_min = 0.1
    
#     control = fl.total_torque(robot, rkd.r0, rkd.rL, rkd.P0, rkd.pm, rkd.rJ, rkd.Rj, rkd.H0, rkd.H0m, link_length, pcd, goal, d_goal, r_min)
    
#     return control

def main():
    rospy.init_node('apf')
    rate = rospy.Rate(10)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # initialize the model and data for mujoco
    # model_path = os.path.join(current_dir, 'model_cjt', 'spacerobot_cjt.xml')
    original_model_path = os.path.join(current_dir, 'model_cjt', 'check.xml')
    model = mujoco.MjModel.from_xml_path(original_model_path)
    data = mujoco.MjData(model)
    
    # initialize the model for spart
    urdf_path = os.path.join(current_dir, 'model_cjt', 'SC_ur10e.urdf')
    robot_spart, robot_key = urdf2robot.urdf2robot(urdf_path)
    rkd = spart.RobotKinematicsDynamics(robot_spart)
    
    stl_path = os.path.join(current_dir, 'model_cjt', 'cuboid_0.5x0.5x2.stl')
    pcd = load_stl_file(stl_path) + np.array([0.0, 1.0, 0.0])
    print("Point cloud loaded with", len(pcd), "points")
    print("Point cloud min:", np.min(pcd, axis=0))
    print("Point cloud max:", np.max(pcd, axis=0))
    # parameters for the APF
    goal = np.array([0.5, 0.5, 0.5])
    
    
    
    if len(pcd) > 1000:  # Adjust this threshold as needed
        # Simple uniform downsampling
        indices = np.random.choice(len(pcd), 1000, replace=False)
        pcd = pcd[indices]
        print("Downsampled to", len(pcd), "points")
    model, _ = add_point_cloud_to_model(original_model_path, pcd, radius=0.01, rgba=[1, 0, 0, 0.8])
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while not rospy.is_shutdown():
            mujoco.mj_step(model, data)
            qpos, qvel = io.get_states(model, data)
            # control = apf_control_value(qpos, qvel, robot_spart, rkd)
            # io.control(model, data, control)
            viewer.sync()
            rate.sleep()



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    