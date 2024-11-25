# Intro:
This package provide a simple Franka arm and Robotiq Gripper simulator written in Mujoco.
It includes a state-based and a vision-based Franka lift cube task environment.

# Installation:
- Fcd into `franka_sim`.
- In your conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Explore the Environments
- Run `python franka_sim/test/test_gym_env_human.py` to launch a display window and visualize the task.
- Run `python franka_sim/test/python xbox_teleoperation.py --gui` to use xbox controller control robot finishing gear assembly in mujoco
- Run `python franka_sim/test/xbox_game_controller.py` to test your xbox controller hardware
- 
# Credits:
- This simulation is initially built by [Kevin Zakka](https://kzakka.com/).


# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
# Franka-Research-3-Robot-Simulation-with-Xbox-Controller-Integration-in-MuJoCo
