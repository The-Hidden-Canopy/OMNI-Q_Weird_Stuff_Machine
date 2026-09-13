#!/usr/bin/env bash
# Detached upload: table_yolo_combined_v1 (6.5GB) -> 4080S box, for the ftv4 round.
# Started via PowerShell Start-Process so it survives the Kimi Code process exiting.
# (retargeted 2026-09-11: old 4060 Ti box 88.90.102.66 was destroyed mid-upload)
set -u
KEY=~/.ssh/id_ed25519
HOST=root@175.155.64.149
PORT=20676
SRC=/e/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/data/table_yolo_combined_v1
DST=/root/omniq/data_upload/table_yolo_combined_v1
LOG=/e/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/tmp/upload_combined_v1.log

mkdir -p "$(dirname "$LOG")"
{
  echo "=== upload start $(date) ==="
  ssh -i "$KEY" -p "$PORT" "$HOST" "mkdir -p /root/omniq/data_upload && rm -f /root/omniq/data_upload/done.flag"
  # no gzip: source is mostly JPEGs, compression buys nothing and burns CPU
  tar cf - -C /e/HiddenCanopy/OMNI-Q_Weird_Stuff_Machine/data table_yolo_combined_v1 \
    | ssh -i "$KEY" -p "$PORT" "$HOST" "tar xf - -C /root/omniq/data_upload && touch /root/omniq/data_upload/done.flag"
  RC=$?
  echo "=== upload end $(date) rc=$RC ==="
  exit $RC
} >> "$LOG" 2>&1
