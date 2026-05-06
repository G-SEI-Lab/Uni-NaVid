#!/usr/bin/env bash
set -u

OUT_DIR="${1:-$HOME/uninavid_diag_$(date +%Y%m%d_%H%M%S)}"
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

run_shell_bg() {
  local name="$1"
  shift
  echo "[monitor] start $name: $*"
  (bash -lc "$*" > "$OUT_DIR/$name.log" 2>&1) &
  echo "$!" >> "$OUT_DIR/pids.txt"
}

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
  rostopic list 2>&1 | grep -E "camera|image|realsense" || true
} > "$OUT_DIR/snapshot_start.txt" 2>&1

if command -v tegrastats >/dev/null 2>&1; then
  tegrastats --interval 1000 --logfile "$OUT_DIR/tegrastats.log" &
  echo "$!" >> "$OUT_DIR/pids.txt"
fi

run_bg vmstat vmstat -t 1
run_shell_bg proc_summary 'while true; do date; free -m; ps -eo pid,ppid,stat,pcpu,pmem,rss,vsz,comm,args --sort=-rss | head -n 30; sleep 1; done'
run_shell_bg disk_io 'while true; do date; cat /proc/diskstats; sleep 2; done'
run_shell_bg kernel_tail 'dmesg -wT'
run_shell_bg ros_node_info 'while true; do date; rosnode list 2>&1; rostopic hz /camera_down/color/image_raw -w 3 2>&1 | head -n 20; sleep 5; done'

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
