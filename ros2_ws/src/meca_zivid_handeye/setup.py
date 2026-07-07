from glob import glob

from setuptools import find_packages, setup

package_name = 'meca_zivid_handeye'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='franzi',
    maintainer_email='franzi@todo.todo',
    description='Automated Zivid eye-to-hand calibration for a fixed camera and Mecademic robot.',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'automated_eye_to_hand = meca_zivid_handeye.automated_eye_to_hand:main',
            'generate_hand_eye_joint_grid = meca_zivid_handeye.generate_hand_eye_joint_grid:main',
        ],
    },
)
