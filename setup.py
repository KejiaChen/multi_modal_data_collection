from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'multi_modal_data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tailai.cheng',
    maintainer_email='tailai.cheng@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multi_sensor_data_recording = multi_modal_data_collection.multi_sensor_data_recording:main',
            'multi_sensor_data_collection = multi_modal_data_collection.multi_sensor_data_collection:main',
            'button_trigger_node = multi_modal_data_collection.button_trigger_node:main',
            'footswitch_trigger_node = multi_modal_data_collection.footswitch_trigger_node:main',
            'multi_sensor_data_collection_with_timestamps = multi_modal_data_collection.multi_sensor_data_collection_with_timestamps:main',
        ],
    },
)
