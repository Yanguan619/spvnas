#!/bin/bash
set +u
cd /data2/jev_work/_fixer
# 等前面的 cr_tiers 跑完
while ps aux | grep -q "[g]m_fg2"; do sleep 20; done
for i in 1 2; do
  echo "########## cr=0.5 aclgraph 复测 #$i"
  TS_GE_COMPAT=1 GM_MODE=npe GM_DYN=0 GM_DEV=0 bash run_model_debug.sh 999999 0.5 5 2>&1 \
    | grep -aE "RESULT" | tail -1
done
echo "### CR05_RECHECK_DONE"
