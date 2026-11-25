#!/bin/bash
set +u
cd /data2/jev_work/_fixer
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export PYTHONPATH="$PYTHONPATH:/data2/jev_work/spvnas-ascend:/data2/jev_work:/data2/jev_work/_fixer"
export UNUM_SD_NOSYNC=1 UNUM_OPS_ROOT=/data2/jev_work/unum_ops_t214
export ASCEND_RT_VISIBLE_DEVICES=0 TS_GE_COMPAT=1
export TS_SGG_MAX_K=100000 TS_SGG_MAX_COUT=512
export GM_MODE=ge GM_DYN=1 GM_DEV=0
for cap in 8192 16384 32768; do
  echo "########## 6k points, CAP=$cap"
  TS_CAP=$cap bash run_model_debug.sh 6000 0.5 5 2>&1 | grep -aE "RESULT" | tail -1
done
echo "### CAPSCALE_DONE"
