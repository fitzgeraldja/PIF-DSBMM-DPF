#!/bin/bash
BASE_DIR=/scratch/fitzgeraldj/data/caus_inf_data
export IN_DIR=${BASE_DIR}
export OUT_DIR=${BASE_DIR}/results
MAIN_REPO_DIR=~/Documents/main_project/post_confirmation/code/PIF-DSBMM-DPF
export MAIN_CODE_DIR=${MAIN_REPO_DIR}/src/pif_dsbmm_dpf

num_procs=8 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running

for SIM_ITER in {0..19};
do
    while ((${num_jobs@P}>=$num_procs)); do
        wait -n
    done
    python ${MAIN_CODE_DIR}/citation/predictive_checks.py \
    --data-dir ${IN_DIR} \
    --out-dir ${OUT_DIR} \
    --seed ${SIM_ITER} &
done
