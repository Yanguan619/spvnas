#!/bin/bash
set +u
cd /data2/jev_work/_fixer
for n in 20000 12000 6000; do
  echo "########## aclgraph npts=$n"
  TS_GE_COMPAT=1 GM_MODE=npe GM_DYN=0 GM_DEV=0 bash run_model_debug.sh $n 0.5 5 2>&1 \
    | grep -aE "RESULT|\[eager\]|\[graph\]" | tail -3
done
echo "### NPE_TIERS_DONE"
