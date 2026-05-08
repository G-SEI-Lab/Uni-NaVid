#!/bin/bash

cd /home/ubuntu/workspace_ros1/cjs_ws
source devel/setup.bash

roslaunch sensor_launcher camera_down.launch
