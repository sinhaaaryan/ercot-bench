#!/usr/bin/env bash
# Memory watchdog for single-box training (built after a host-RAM OOM took down the whole WSL VM).
# Every INTERVAL seconds it checks host MemAvailable and GPU memory.used, and for all running ercot-*.scope units:
#   - RAM available < PAUSE_MB   -> freeze (pause) all ercot scopes
#   - once RAM > RESUME_MB        -> thaw the scopes it paused
#   - RAM available < KILL_MB     -> stop the ercot scope using the most RAM
#   - VRAM used > GPU_PAUSE_MB    -> log a warning only. A CUDA OOM fails just that process (it cannot take down the
#     machine), and freezing a job does not release its VRAM, so a GPU-triggered pause would never resume.
# Defaults assume a 40 GB RAM / 32 GB VRAM machine. Log: $LOG. Single instance via $PIDFILE.
PAUSE_MB=${PAUSE_MB:-6000}; RESUME_MB=${RESUME_MB:-9000}; KILL_MB=${KILL_MB:-2500}
GPU_PAUSE_MB=${GPU_PAUSE_MB:-31800}; GPU_RESUME_MB=${GPU_RESUME_MB:-31000}
INTERVAL=${INTERVAL:-2}; LOG=${LOG:-/hackathon/mem_guard.log}; PIDFILE=${PIDFILE:-/tmp/ercot_mem_guard.pid}
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "mem_guard already running (pid $(cat "$PIDFILE"))"; exit 0; fi
echo $$ > "$PIDFILE"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
scopes() { systemctl --user list-units --type=scope --state=running --no-legend 'ercot-*' 2>/dev/null | awk '{print $1}'; }
paused=0
log "mem_guard started: pause<${PAUSE_MB}MB resume>${RESUME_MB}MB kill<${KILL_MB}MB gpu_pause>${GPU_PAUSE_MB}MB"
while true; do
  avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
  gpu=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9); gpu=${gpu:-0}
  if [ "$avail" -lt "$KILL_MB" ]; then
    big=$(for s in $(scopes); do echo "$(systemctl --user show "$s" -p MemoryCurrent --value) $s"; done | sort -n | tail -1 | awk '{print $2}')
    [ -n "$big" ] && { log "CRITICAL ram_avail=${avail}MB -> stopping $big"; systemctl --user stop "$big"; }
  elif [ "$paused" -eq 0 ] && [ "$avail" -lt "$PAUSE_MB" ]; then
    for s in $(scopes); do systemctl --user freeze "$s" && log "PAUSED $s (ram_avail=${avail}MB gpu_used=${gpu}MB)"; done
    paused=1
  elif [ "$paused" -eq 1 ] && [ "$avail" -gt "$RESUME_MB" ]; then
    for s in $(scopes); do systemctl --user thaw "$s" && log "RESUMED $s (ram_avail=${avail}MB gpu_used=${gpu}MB)"; done
    paused=0
  fi
  if [ "$gpu" -gt "$GPU_PAUSE_MB" ]; then
    now=$(date +%s); if [ $(( now - ${last_gpu_warn:-0} )) -ge 60 ]; then log "WARN gpu_used=${gpu}MB (> ${GPU_PAUSE_MB}MB)"; last_gpu_warn=$now; fi
  fi
  sleep "$INTERVAL"
done
