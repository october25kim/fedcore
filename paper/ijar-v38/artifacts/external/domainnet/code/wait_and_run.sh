#!/usr/bin/env bash
# Waits until ubuntu GPU0 has no compute process on 2 consecutive polls (30 s apart), then runs drycheck -> infer -> audit -> analyze.
set -uo pipefail
F=/home/sanghoon/Desktop/Workspace/Fedcore; R=$F/outputs/fedcore_s1v2_run_20260925; S=$F/work/s1v2/run_stage.sh
L=$R/LAUNCH_LOG.txt
echo "[$(date "+%F %T %Z")] watcher start" >> $L
ok=0
while true; do
  n=$(nvidia-smi --id=0 --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  if [ "$n" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 2 ]; then break; fi
  sleep 30
done
echo "[$(date "+%F %T %Z")] GPU0 idle confirmed:" >> $L; nvidia-smi --id=0 --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader >> $L
echo "[$(date "+%F %T %Z")] drycheck start" >> $L
$S drycheck >> $R/drycheck_stdout.txt 2>&1 || { echo "[$(date "+%F %T %Z")] DRYCHECK FAIL - stop" >> $L; exit 3; }
echo "[$(date "+%F %T %Z")] drycheck PASS; infer start" >> $L; nvidia-smi --id=0 --query-compute-apps=pid,used_memory --format=csv,noheader >> $L
T0=$(date +%s)
$S infer --workers 6 >> $R/infer_stdout.txt 2>&1 || { echo "[$(date "+%F %T %Z")] INFER FAIL/STOP rc=$?" >> $L; exit 4; }
echo "[$(date "+%F %T %Z")] infer done wall=$(( $(date +%s)-T0 ))s; audit start (CPU)" >> $L
$S audit >> $R/audit_stdout.txt 2>&1 || { echo "[$(date "+%F %T %Z")] AUDIT FAIL" >> $L; exit 5; }
echo "[$(date "+%F %T %Z")] audit done; analyze start (CPU)" >> $L
$S analyze >> $R/analyze_stdout.txt 2>&1 || { echo "[$(date "+%F %T %Z")] ANALYZE rc=$?" >> $L; exit 6; }
echo "[$(date "+%F %T %Z")] ALL DONE" >> $L
