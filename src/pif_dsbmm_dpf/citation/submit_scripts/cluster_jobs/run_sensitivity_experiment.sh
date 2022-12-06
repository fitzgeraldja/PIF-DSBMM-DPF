#!/bin/bash
#SBATCH -A sml
#SBATCH --mem-per-cpu=64gb
source activate influence
python -m pokec.run_sensitivity_study \
--data_dir=${IN_DIR} \
--out_dir=${OUT_DIR} \
--num_components=${NUM_COMPONENTS} \
--num_exog_components=${NUM_EXOG_COMPONENTS} \
--seed=${SEED}
