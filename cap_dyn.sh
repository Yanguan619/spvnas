#!/bin/bash
set +u
cd /data2/jev_work/_fixer
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
export PYTHONPATH="$PYTHONPATH:/data2/jev_work/spvnas-ascend:/data2/jev_work:/data2/jev_work/_fixer"
export UNUM_SD_NOSYNC=1 UNUM_OPS_ROOT=/data2/jev_work/unum_ops_t214
export ASCEND_RT_VISIBLE_DEVICES=0 TS_GE_COMPAT=1
export TS_SGG_MAX_K=100000 TS_SGG_MAX_COUT=512
export GM_MODE=ge GM_DYN=1 GM_DEV=0

echo "########## 94k cr=0.25 CAP=131072"
TS_CAP=131072 bash run_model_debug.sh 999999 0.25 5 2>&1 | grep -aE "RESULT" | tail -1
echo "########## 94k cr=0.5 CAP=131072"
TS_CAP=131072 bash run_model_debug.sh 999999 0.5 5 2>&1 | grep -aE "RESULT" | tail -1
echo "########## 94k cr=1.0 CAP=131072"
TS_CAP=131072 bash run_model_debug.sh 999999 1.0 5 2>&1 | grep -aE "RESULT" | tail -1
echo "########## 20k cr=0.5 CAP=20480"
TS_CAP=20480 bash run_model_debug.sh 20000 0.5 5 2>&1 | grep -aE "RESULT" | tail -1
echo "########## 12k cr=0.5 CAP=16384"
TS_CAP=16384 bash run_model_debug.sh 12000 0.5 5 2>&1 | grep -aE "RESULT" | tail -1
echo "########## 6k cr=0.5 CAP=8192"
TS_CAP=8192 bash run_model_debug.sh 6000 0.5 5 2>&1 | grep -aE "RESULT" | tail -1
echo "### CAPDYN_DONE"
