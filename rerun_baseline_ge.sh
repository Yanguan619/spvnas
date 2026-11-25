#!/bin/bash
set +u
cd /data2/jev_work/_fixer
echo "########## GE 静态 + TS_GE_COMPAT=1"
TS_GE_COMPAT=1 GM_MODE=ge GM_DYN=0 GM_DEV=0 bash run_model_debug.sh 999999 0.5 3 2>&1 | tail -25
echo "########## GE 动态 + TS_GE_COMPAT=1"
TS_GE_COMPAT=1 GM_MODE=ge GM_DYN=1 GM_DEV=0 bash run_model_debug.sh 999999 0.5 3 2>&1 | tail -25
echo "### RERUN4_DONE"
