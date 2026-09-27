#!/usr/bin/env bash
# Run a command inside its own systemd --user scope with a hard RAM cap, so an OOM kills only this job
# (not the whole WSL VM). Pair with scripts/mem_guard.sh, which pauses/resumes ercot-* scopes on low RAM/VRAM.
#
#   scripts/guarded_run.sh <name> <MemoryMax, e.g. 18G> <command...>
#
# The scope is ercot-<name>.scope; inspect with `systemctl --user status ercot-<name>.scope`,
# pause/resume with `systemctl --user freeze|thaw ercot-<name>.scope`, stop with `systemctl --user stop ...`.
set -euo pipefail
name="$1"; max="$2"; shift 2
num="${max%[GgMm]}"; unit="${max: -1}"
high="$(( num * 9 / 10 ))${unit}"   # start reclaiming/throttling at 90% of the cap
exec systemd-run --user --scope --quiet --unit="ercot-${name}" \
  -p MemoryMax="$max" -p MemoryHigh="$high" -p MemorySwapMax=0 "$@"
