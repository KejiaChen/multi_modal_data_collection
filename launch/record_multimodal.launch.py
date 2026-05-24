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
        DeclareLaunchArgument('enable_rgb', default_value='true'),
        DeclareLaunchArgument('enable_rgb2', default_value='false'),
        DeclareLaunchArgument('enable_vive', default_value='true'),
        DeclareLaunchArgument('enable_vive_ultimate', default_value='true'),
        DeclareLaunchArgument('enable_tactile', default_value='true'),
        DeclareLaunchArgument('rgb_topic', default_value='/camera_up/color/image_rect_raw'),
        DeclareLaunchArgument('rgb2_topic', default_value='/camera_down/color/image_rect_raw'),
        DeclareLaunchArgument('tactile_topic', default_value='/gelsight/image_raw'),
        DeclareLaunchArgument('vive_topic', default_value='/vive_tracker/pose'),
        DeclareLaunchArgument('vive_ultimate_topic', default_value='/vive_ultimate_tracker/pose'),        # --- node launch ---

        # --- node launch ---
        Node(
            package='multi_modal_data_collection',
            executable='multi_sensor_data_collection',
            name='multi_sensor_data_collection',
            output='screen',
            parameters=[
                {
                    'out_dir': LaunchConfiguration('out_dir'),
                    'rate_hz': LaunchConfiguration('rate_hz'),
                    'slop_sec': LaunchConfiguration('slop_sec'),
                    'enable_rgb': LaunchConfiguration('enable_rgb'),
                    'enable_rgb2': LaunchConfiguration('enable_rgb2'),
                    'enable_vive': LaunchConfiguration('enable_vive'),
                    'enable_vive_ultimate': LaunchConfiguration('enable_vive_ultimate'),
                    'enable_tactile': LaunchConfiguration('enable_tactile'),
                    'rgb_topic': LaunchConfiguration('rgb_topic'),
                    'rgb2_topic': LaunchConfiguration('rgb2_topic'),
                    'tactile_topic': LaunchConfiguration('tactile_topic'),
                    'vive_topic': LaunchConfiguration('vive_topic'),                }
            ]
        )
    ])
