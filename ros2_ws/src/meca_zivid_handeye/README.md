# Meca Zivid Hand-Eye Calibration

Automated eye-to-hand calibration for a fixed Zivid camera and a Mecademic/Meca robot.

This package provides to functionalities:
1. Create a set of configurations automatically (can be used for convenience, but can also be created manually)
2. Move the robot through a list of joint configurations, captures the Zivid calibration board at each reachable/visible pose, keeps only valid detections, selects a diverse subset of capture-pose pairs, and calls the Zivid ROS hand-eye calibration service.

## Assumptions

- ROS 2 Humble workspace.
- Zivid SDK and `zivid-ros` packages are already installed in the same workspace.
- `mecademicpy` is installed on the robot-control machine.
- The Zivid camera is fixed in the workspace.
- The official Zivid calibration board is rigidly mounted to the robot flange/tool.
- Meca `GetPose()` returns the flange pose in robot base frame.
- Robot positions are in millimeters.
- Meca orientation is intrinsic XYZ Euler angles. The script converts this to ROS quaternion `x, y, z, w`.

The generated hand-eye transform is the camera pose in robot base frame:

```text
parent: meca_base_link
child:  pcl_frame
```

For a point in camera coordinates:

```text
p_meca_base = T_meca_base_camera * p_camera
```

## Safety

Pay close attention while moving the robot. The generator only checks numeric joint limits; it does not know about (self-) collisions, singularities, or the current workspace.

Keyboard controls are convenience controls only:

- `SPACE`: skip the current pose and continue.
- `ENTER`: stop collecting more poses and calibrate with valid candidates collected so far.

Use the robot emergency stop for real collision risk.

## Generate Joint Configurations
```
ros2 run meca_zivid_handeye generate_hand_eye_joint_grid \
  --output ~/ws_neon/src/meca_zivid_handeye/config/hand_eye_joint_poses.center_limited.generated.json \
  --center "68.64078,47.97905,19.04922,-91.26776,-95.2169,31.0069" \
  --joint-limits="-170,170;-65,85;-130,65;-165,165;-110,110;-180,180" \
  --max-count 100 \
  --j1-offsets=-30,-15,0,15,30 \
  --j2-offsets=-12,0,12 \
  --j3-offsets=-12,0,12 \
  --j4-offsets=-45,0,45 \
  --j5-offsets=-8,0,8,18 \
  --j6-offsets=-60,-30,0,30,60
```

This creates a joint-space grid around the visible center pose, filters by the provided joint limits, selects a diverse subset, and orders the output by nearest-neighbor joint motion. The center is a configuration in which the callibration board is clearly visible to the camera and the next potential collision is far away.

`--max-count` describes how many configurations should be generated.

## Run Calibration

Terminal 1: start the Zivid camera node and keep it running.

```bash
ros2 run zivid_camera zivid_camera
```
Terminal 2: run the automated calibration.

```bash
ros2 run meca_zivid_handeye automated_eye_to_hand --ros-args \
  -p robot_ip:=192.168.0.100 \
  -p joint_configurations_file:=~/ws_neon/src/meca_zivid_handeye/config/<path to hand_eye_joint_poses.center_limited.generated.json> \
  -p min_successful_captures:=6 \
  -p selected_capture_count:=20
```

During the run:

- Poses with no calibration-board detection are skipped.
- Duplicate robot poses are skipped.
- Valid board detections become calibration candidates.
- The node selects the most diverse `selected_capture_count` candidates before calibration.

## Outputs

Each run writes to:

```text
/tmp/meca_zivid_handeye_<timestamp>/tf2_zivid_robotBase.generated.yaml
```
