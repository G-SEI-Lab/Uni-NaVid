#!/usr/bin/env bash
set -euo pipefail

TOPIC="/uninavid/instruction"
MSG_TYPE="std_msgs/String"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 \"move forward to the target and stop.\""
  exit 1
fi

# 把所有参数合并成一个字符串
MSG="$*"

# 用 JSON/YAML 安全转义，避免字符串里有引号、冒号等导致 rostopic 解析失败
QUOTED_MSG=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$MSG")

rostopic pub --once "$TOPIC" "$MSG_TYPE" -- "{data: $QUOTED_MSG}"
