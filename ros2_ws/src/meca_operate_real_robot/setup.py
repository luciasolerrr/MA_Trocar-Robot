from setuptools import find_packages, setup

package_name = 'meca_operate_real_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='franzi',
    maintainer_email='franzi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'convert_to_meca_node = meca_operate_real_robot.convert_to_meca_node:main',
        'send_joint_sequence_node = meca_operate_real_robot.send_joint_sequence_node:main',
        'send_cartesian_sequence_node = meca_operate_real_robot.send_cartesian_sequence_node:main',
        'tf_eef_base_node = meca_operate_real_robot.tf_eef_base_node:main'
        ],
    },
)
