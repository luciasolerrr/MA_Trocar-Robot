# MA_Trocar-Robot
````markdown
# How to Use

## 1. Setup

Clone the repository:

```bash
git clone https://github.com/luciasolerrr/MA_Trocar-Robot.git
````

These instructions assume that `TFM` and `ros2_ws` are located in the Home directory.

Create the Python environment:

```bash
cd ~/TFM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Build the ROS 2 workspace:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Before running the pipeline, check that these files are available:

```text
TFM/models/model_A_yolo26s.pt
TFM/models/model_b_letterbox_best.pt
TFM/calibration/tf2_zivid_robotBase.generated.yaml
TFM/ZIVID/Z2p_M60_Inspection_SmallFeatures_Off.yml
```

---

## 2. Run the System

### Terminal 1 – Connect to the Meca500

```bash
cd ~/ros2_ws
source install/setup.bash

ros2 launch meca_with_needle_moveit_config \
    demo.launch_real_meca.py use_rviz:=false
```

### Terminal 2 – RViz

```bash
source ~/ros2_ws/install/setup.bash

ros2 launch meca_with_needle_moveit_config \
    rviz_only.launch.py
```

### Terminal 3 – Run the inference pipeline

```bash
cd ~/ros2_ws
source install/setup.bash

cd ~/TFM
source .venv/bin/activate

python pipeline_inference_direct_movepose.py
```

The script guides the user through the complete procedure:

1. Select the eye side.
2. Enter the test name.
3. Confirm the initial safe robot pose.
4. Capture the RGB-D data.
5. Detect the eye and segment the trocar.
6. Estimate the trocar pose.
7. Move the robot to the approach pose.
8. Confirm the linear approach.
9. Adjust J6 manually if required.
10. Perform the final insertion.

Results are saved in:

```text
TFM/Inference/<case_name>/
```

---

## 3. Main Scripts

```text
TFM/pipeline_inference_direct_movepose_final.py
    Complete inference and robot execution pipeline.

TFM/pipeline_inference_letterbox_z.py
    Alternative pipeline using MoveIt.

TFM/pose_utils_trocarfit.py
    Eye sphere fitting and trocar pose estimation.

TFM/build_meca_pose_from_axis_z.py
    Converts the estimated target to the Meca500 base frame.

ros2_ws/scripts/rotate_j6_interactive.py
    Manual J6 alignment.

ros2_ws/scripts/move_lin_advance_current.py
    Linear TCP advance.
```


