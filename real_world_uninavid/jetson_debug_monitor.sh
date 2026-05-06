#!/usr/bin/env bash
set -u

OUT_DIR="${1:-$HOME/uninavid_diag_$(date +%Y%m%d_%H%M%S)}"
MONITOR_CAMERA_TOPIC="${MONITOR_CAMERA_TOPIC:-/camera_down/color/image_raw}"
mkdir -p "$OUT_DIR"

echo "[monitor] writing logs to $OUT_DIR"
echo "$OUT_DIR" > "$OUT_DIR/OUT_DIR.txt"

run_bg() {
  local name="$1"
  shift
  echo "[monitor] start $name: $*"
  ("$@" > "$OUT_DIR/$name.log" 2>&1) &
  echo "$!" >> "$OUT_DIR/pids.txt"
}

can_sudo_nopass() {
  sudo -n true >/dev/null 2>&1
}

start_kernel_tail() {
  if [ "$(id -u)" -eq 0 ]; then
    run_shell_bg kernel_tail "dmesg -wT"
    return
  fi
  if can_sudo_nopass; then
    run_shell_bg kernel_tail "sudo -n dmesg -wT"
    return
  fi
  echo "[monitor] WARN: cannot read kernel ring buffer without root or passwordless sudo" | tee "$OUT_DIR/kernel_tail.log"
  echo "[monitor] WARN: run monitor as root or configure sudo -n for dmesg" >> "$OUT_DIR/kernel_tail.log"
}

run_shell_bg() {
  local name="$1"
  shift
  echo "[monitor] start $name: $*"
  (bash -lc "$*" > "$OUT_DIR/$name.log" 2>&1) &
  echo "$!" >> "$OUT_DIR/pids.txt"
}

init_ros_env() {
  local sourced_any=0
  local ros_setup_file="${ROS_SETUP_FILE:-}"
  local ros_ws_setup_file="${ROS_WS_SETUP_FILE:-}"
  local ws_file

  if [ -n "$ros_setup_file" ] && [ -r "$ros_setup_file" ]; then
    # shellcheck disable=SC1090
    source "$ros_setup_file"
    sourced_any=1
    echo "[monitor] sourced ROS setup: $ros_setup_file"
  elif [ -r "/opt/ros/noetic/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "/opt/ros/noetic/setup.bash"
    sourced_any=1
    echo "[monitor] sourced ROS setup: /opt/ros/noetic/setup.bash"
  fi

  if [ -n "$ros_ws_setup_file" ] && [ -r "$ros_ws_setup_file" ]; then
    # shellcheck disable=SC1090
    source "$ros_ws_setup_file"
    sourced_any=1
    echo "[monitor] sourced ROS workspace setup: $ros_ws_setup_file"
  else
    for ws_file in \
      "$PWD/devel/setup.bash" \
      "$PWD/../devel/setup.bash" \
      "$HOME/workspace_ros1/devel/setup.bash" \
      "$HOME/catkin_ws/devel/setup.bash"
    do
      if [ -r "$ws_file" ]; then
        # shellcheck disable=SC1090
        source "$ws_file"
        sourced_any=1
        echo "[monitor] sourced ROS workspace setup: $ws_file"
        break
      fi
    done
  fi

  if command -v rostopic >/dev/null 2>&1; then
    echo "[monitor] rostopic found: $(command -v rostopic)"
  else
    echo "[monitor] WARN: rostopic not found; set ROS_SETUP_FILE or ROS_WS_SETUP_FILE before running monitor"
  fi

  if [ "$sourced_any" -eq 0 ]; then
    echo "[monitor] WARN: no ROS setup script sourced"
  fi
}

init_ros_env > "$OUT_DIR/ros_env.log" 2>&1 || true

{
  date
  uname -a
  echo "---- nvpmodel ----"
  nvpmodel -q 2>&1 || true
  echo "---- jetson clocks ----"
  jetson_clocks --show 2>&1 || true
  echo "---- disk ----"
  df -h
  df -ih
  echo "---- memory ----"
  free -h
  swapon --show || true
  echo "---- usb ----"
  lsusb -t 2>&1 || true
  echo "---- camera topics ----"
  if command -v rostopic >/dev/null 2>&1; then
    rostopic list 2>&1 | grep -E "camera|image|realsense" || true
  else
    echo "rostopic not found"
  fi
} > "$OUT_DIR/snapshot_start.txt" 2>&1

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 1000 --logfile "$OUT_DIR/tegrastats.log" &
  echo "$!" >> "$OUT_DIR/pids.txt"
fi

run_bg vmstat vmstat -t 1
run_shell_bg proc_summary 'while true; do date; free -m; ps -eo pid,ppid,stat,pcpu,pmem,rss,vsz,comm,args --sort=-rss | head -n 30; sleep 1; done'
run_shell_bg disk_io 'while true; do date; cat /proc/diskstats; sleep 2; done'
start_kernel_tail
run_shell_bg ros_node_info "while true; do date; if command -v rosnode >/dev/null 2>&1; then rosnode list 2>&1; else echo '[monitor] rosnode command not found'; fi; if command -v rostopic >/dev/null 2>&1; then rostopic hz \"$MONITOR_CAMERA_TOPIC\" -w 3 2>&1 | head -n 20; else echo '[monitor] rostopic command not found'; fi; sleep 5; done"

cat > "$OUT_DIR/stop_monitor.sh" <<'EOF'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$DIR/pids.txt" ]; then
  xargs -r kill < "$DIR/pids.txt" 2>/dev/null || true
fi
pkill -f "tegrastats --interval 1000 --logfile $DIR/tegrastats.log" 2>/dev/null || true
echo "[monitor] stopped"
EOF
chmod +x "$OUT_DIR/stop_monitor.sh"

echo "[monitor] started. Stop with: $OUT_DIR/stop_monitor.sh"
echo "[monitor] after reboot, also collect:"
echo "  journalctl -k -b -1 -o short-precise > $OUT_DIR/kernel_prev_boot.log"
echo "  dmesg -T > $OUT_DIR/dmesg_after_reboot.log"

while true; do
  sleep 3600
done
