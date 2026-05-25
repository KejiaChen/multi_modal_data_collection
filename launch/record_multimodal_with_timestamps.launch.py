from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # --- configurable parameters ---
        DeclareLaunchArgument('out_dir', default_value='/home/tailai.cheng/tailai_ws/src/multi_modal_data_collection/data'),
        DeclareLaunchArgument('rate_hz', default_value='10.0'),
        DeclareLaunchArgument('slop_sec', default_value='0.1'),
        DeclareLaunchArgument('enable_rgb', default_value='false'),
        DeclareLaunchArgument('enable_rgb2', default_value='false'),
        DeclareLaunchArgument('enable_vive', default_value='false'),
        DeclareLaunchArgument('enable_ultimate_vive', default_value='false'),
        DeclareLaunchArgument('enable_tactile', default_value='false'),
        DeclareLaunchArgument('enable_franka_ee_pose', default_value='false'),
        DeclareLaunchArgument('enable_franka_ee_pose_cmd', default_value='false'),
        DeclareLaunchArgument('enable_franka_gripper_width', default_value='false'),
        DeclareLaunchArgument('enable_franka_gripper_grasp', default_value='false'),

        DeclareLaunchArgument('rgb_topic', default_value='/camera_up/color/image_rect_raw'),
        DeclareLaunchArgument('rgb2_topic', default_value='/camera_down/color/image_rect_raw'),
        DeclareLaunchArgument('rgb_resize_width', default_value='0'),
        DeclareLaunchArgument('rgb_resize_height', default_value='0'),
        DeclareLaunchArgument('rgb2_resize_width', default_value='0'),
        DeclareLaunchArgument('rgb2_resize_height', default_value='0'),
        DeclareLaunchArgument('tactile_topic', default_value='/gelsight/image_raw'),
        DeclareLaunchArgument('vive_topic', default_value='/vive_tracker/pose'),
        DeclareLaunchArgument('ultimate_vive_topic', default_value='/vive_ultimate_tracker/pose'),
        DeclareLaunchArgument('franka_ee_pose_topic', default_value='/frankaRight/ee_pose'),
        DeclareLaunchArgument('franka_ee_pose_cmd_topic', default_value='/frankaRight/ee_pose_cmd'),
        DeclareLaunchArgument('franka_gripper_width_topic', default_value='/frankaRight/gripper_width'),
        DeclareLaunchArgument('franka_gripper_grasp_topic', default_value='/frankaRight/is_grasped'),
        # -------------------------------
        # LUCID camera parameters 
        # -------------------------------
        DeclareLaunchArgument('enable_lucid', default_value='false'),
        DeclareLaunchArgument('lucid_topic', default_value='/rgb_lucid'),
        DeclareLaunchArgument('save_format', default_value='npz'),
        DeclareLaunchArgument('robot_type', default_value='custom'),
        DeclareLaunchArgument('task_name', default_value='default_task'),
        DeclareLaunchArgument('lerobot_chunk_size', default_value='1000'),

        # --- NEW node launch (timestamps version) ---
        Node(
            package='multi_modal_data_collection',
            executable='multi_sensor_data_collection_with_timestamps',
            name='multi_sensor_data_collection_with_timestamps',
            output='screen',
            parameters=[
                {
                    'out_dir': LaunchConfiguration('out_dir'),
                    'rate_hz': LaunchConfiguration('rate_hz'),
                    'slop_sec': LaunchConfiguration('slop_sec'),
                    'enable_rgb': LaunchConfiguration('enable_rgb'),
                    'enable_rgb2': LaunchConfiguration('enable_rgb2'),
                    'enable_vive': LaunchConfiguration('enable_vive'),
                    'enable_vive_ultimate': LaunchConfiguration('enable_ultimate_vive'),
                    'enable_tactile': LaunchConfiguration('enable_tactile'),
                    'enable_franka_ee_pose': LaunchConfiguration('enable_franka_ee_pose'),
                    'enable_franka_ee_pose_cmd': LaunchConfiguration('enable_franka_ee_pose_cmd'),
                    'enable_franka_gripper_width': LaunchConfiguration('enable_franka_gripper_width'),
                    'enable_franka_gripper_grasp': LaunchConfiguration('enable_franka_gripper_grasp'),
                    'rgb_topic': LaunchConfiguration('rgb_topic'),
                    'rgb2_topic': LaunchConfiguration('rgb2_topic'),
                    'rgb_resize_width': LaunchConfiguration('rgb_resize_width'),
                    'rgb_resize_height': LaunchConfiguration('rgb_resize_height'),
                    'rgb2_resize_width': LaunchConfiguration('rgb2_resize_width'),
                    'rgb2_resize_height': LaunchConfiguration('rgb2_resize_height'),
                    'tactile_topic': LaunchConfiguration('tactile_topic'),
                    'vive_topic': LaunchConfiguration('vive_topic'),
                    'vive_ultimate_topic': LaunchConfiguration('ultimate_vive_topic'),
                    'franka_ee_pose_topic': LaunchConfiguration('franka_ee_pose_topic'),
                    'franka_ee_pose_cmd_topic': LaunchConfiguration('franka_ee_pose_cmd_topic'),
                    'franka_gripper_width_topic': LaunchConfiguration('franka_gripper_width_topic'),
                    'franka_gripper_grasp_topic': LaunchConfiguration('franka_gripper_grasp_topic'),
                    # LUCID camera
                    'enable_lucid': LaunchConfiguration('enable_lucid'),
                    'lucid_topic':  LaunchConfiguration('lucid_topic'),
                    'save_format': LaunchConfiguration('save_format'),
                    'robot_type': LaunchConfiguration('robot_type'),
                    'task_name': LaunchConfiguration('task_name'),
                    'lerobot_chunk_size': LaunchConfiguration('lerobot_chunk_size'),
                
                }
            ]
        )
    ])
