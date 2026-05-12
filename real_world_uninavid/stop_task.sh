#!/bin/bash

rostopic pub --once /uninavid/cancel std_msgs/Bool "data: true"

