#!/bin/bash
set +u
cd /data2/jev_work/_fixer
for cr in 0.25 1.0; do
  echo "########## cr=$cr aclgraph"
  TS_GE_COMPAT=1 GM_MODE=npe GM_DYN=0 GM_DEV=0 bash run_model_debug.sh 999999 $cr 5 2>&1 \
    | grep -aE "RESULT" | tail -1
  echo "########## cr=$cr GE静态"
  TS_GE_COMPAT=1 GM_MODE=ge GM_DYN=0 GM_DEV=0 bash run_model_debug.sh 999999 $cr 5 2>&1 \
    | grep -aE "RESULT" | tail -1
  echo "########## cr=$cr GE动态"
  TS_GE_COMPAT=1 GM_MODE=ge GM_DYN=1 GM_DEV=0 bash run_model_debug.sh 999999 $cr 5 2>&1 \
    | grep -aE "RESULT" | tail -1
done
echo "### CR_TIERS_DONE"
