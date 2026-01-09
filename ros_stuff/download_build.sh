source /opt/ros/jazzy/setup.bash
cd ./ros_ws/
mkdir src
vcs import src < dependencies.yaml
colcon build
source ./install/setup.bash