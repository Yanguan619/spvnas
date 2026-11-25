#!/bin/bash
set +u
cd /data2/jev_work/_fixer
echo "########## eager（GM_MODE=npe，全帧）"
GM_MODE=npe GM_DEV=0 bash run_model_debug.sh 999999 0.5 3 2>&1 | grep -aE "RESULT|eager|\[eager\]" | tail -3
echo "########## GE 静态（GM_DYN=0，全帧）"
GM_MODE=ge GM_DYN=0 GM_DEV=0 bash run_model_debug.sh 999999 0.5 3 2>&1 | grep -aE "RESULT|\[fg\]" | tail -3
echo "########## GE 动态（GM_DYN=1，全帧）"
GM_MODE=ge GM_DYN=1 GM_DEV=0 bash run_model_debug.sh 999999 0.5 3 2>&1 | grep -aE "RESULT|\[fg\]" | tail -3
echo "### RERUN3_DONE"
