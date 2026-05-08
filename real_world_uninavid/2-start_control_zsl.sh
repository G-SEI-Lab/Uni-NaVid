#!/usr/bin/bash

docker start zsl_server
sleep 3
python3 /home/ubuntu/workspace_ros1/elevator-robot/scripts/zsibot/client_in_orin.py
