from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # --- configurable parameters ---
        DeclareLaunchArgument('out_dir', default_value='/home/agile/ros2_ws/src/multi_modal_data_collection/data'),
        DeclareLaunchArgument('rate_hz', default_value='10.0'),
        DeclareLaunchArgument('slop_sec', default_value='0.1'),
        DeclareLaunchArgument('enable_rgb', default_value='true'),
        DeclareLaunchArgument('enable_rgb2', default_value='false'),
        DeclareLaunchArgument('enable_vive', default_value='false'),
        DeclareLaunchArgument('enable_ultimate_vive', default_value='true'),
        # DeclareLaunchArgument('enable_tactile', default_value='true'),

        # ---- Tactile (GelSight) left / right ----
        DeclareLaunchArgument('enable_tactile_left',  default_value='false'),
        DeclareLaunchArgument('enable_tactile_right', default_value='false'),

        DeclareLaunchArgument('rgb_topic', default_value='/camera/camera_up/color/image_rect_raw'),
        DeclareLaunchArgument('rgb2_topic', default_value='/camera/camera_down/color/image_rect_raw'),
        # DeclareLaunchArgument('tactile_topic', default_value='/gelsight/image_raw'),
        DeclareLaunchArgument('tactile_left_topic',  default_value='/gelsight/left/image_raw'),
        DeclareLaunchArgument('tactile_right_topic', default_value='/gelsight/right/image_raw'),
        DeclareLaunchArgument('vive_topic', default_value='/vive_tracker/pose'),
        DeclareLaunchArgument('ultimate_vive_topic', default_value='/vive_ultimate_tracker/pose'),

        # -------------------------------
        # Manus glove parameters
        # -------------------------------
        DeclareLaunchArgument('enable_manus_right_ergo', default_value='true'),
        DeclareLaunchArgument('enable_manus_left_ergo',  default_value='false'),
        DeclareLaunchArgument('enable_manus_right_nodes', default_value='true'),
        DeclareLaunchArgument('enable_manus_left_nodes',  default_value='false'),

        # DeclareLaunchArgument('manus_right_topic', default_value='/manus_glove_right_corrected'),
        DeclareLaunchArgument('manus_right_topic', default_value='/manus_glove_0'),
        # DeclareLaunchArgument('manus_left_topic',  default_value='/manus_glove_left_corrected'),
        DeclareLaunchArgument('manus_left_topic', default_value='/manus_glove_1'),

        DeclareLaunchArgument('enable_manus_right_palm_pose', default_value='true'),
        DeclareLaunchArgument('enable_manus_left_palm_pose',  default_value='false'),
        DeclareLaunchArgument('manus_right_palm_pose_topic', default_value='/manus/right_palm/pose'),
        DeclareLaunchArgument('manus_left_palm_pose_topic',  default_value='/manus/left_palm/pose'),

        # -------------------------------
        # LUCID camera parameters 
        # -------------------------------
        DeclareLaunchArgument('enable_lucid', default_value='false'),
        DeclareLaunchArgument('lucid_topic', default_value='/rgb_lucid'),

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
                    'enable_ultimate_vive': LaunchConfiguration('enable_ultimate_vive'),
                    # 'enable_tactile': LaunchConfiguration('enable_tactile'),
                    'enable_tactile_left': LaunchConfiguration('enable_tactile_left'),
                    'enable_tactile_right': LaunchConfiguration('enable_tactile_right'),
                    'rgb_topic': LaunchConfiguration('rgb_topic'),
                    'rgb2_topic': LaunchConfiguration('rgb2_topic'),
                    # 'tactile_topic': LaunchConfiguration(ros2 run manus_glove_viz manus_glove_visualizer --ros-args -p mount:=left'tactile_topic'),
                    'tactile_left_topic': LaunchConfiguration('tactile_left_topic'),
                    'tactile_right_topic': LaunchConfiguration('tactile_right_topic'),
                    'vive_topic': LaunchConfiguration('vive_topic'),
                    'ultimate_vive_topic': LaunchConfiguration('ultimate_vive_topic'),

                    # manus glove
                    'enable_manus_right_ergo': LaunchConfiguration('enable_manus_right_ergo'),
                    'enable_manus_left_ergo':  LaunchConfiguration('enable_manus_left_ergo'),
                    'enable_manus_right_nodes': LaunchConfiguration('enable_manus_right_nodes'),
                    'enable_manus_left_nodes':  LaunchConfiguration('enable_manus_left_nodes'),

                    'manus_right_topic': LaunchConfiguration('manus_right_topic'),
                    'manus_left_topic':  LaunchConfiguration('manus_left_topic'),

                    'enable_manus_right_palm_pose': LaunchConfiguration('enable_manus_right_palm_pose'),
                    'enable_manus_left_palm_pose':  LaunchConfiguration('enable_manus_left_palm_pose'),
                    'manus_right_palm_pose_topic': LaunchConfiguration('manus_right_palm_pose_topic'),
                    'manus_left_palm_pose_topic':  LaunchConfiguration('manus_left_palm_pose_topic'),

                    # LUCID camera
                    'enable_lucid': LaunchConfiguration('enable_lucid'),
                    'lucid_topic':  LaunchConfiguration('lucid_topic'),
                
                }
            ]
        )
    ])
