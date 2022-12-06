#!/bin/bash

python -m citation.run_experiment \
--data_dir=${DIR} \
--out_dir=${OUT} \
--model=${MODEL} \
--variant=${VARIANT} \
--num_components=${NUM_COMPONENTS} \
--num_exog_components=${NUM_EXOG_COMPONENTS} \
--confounding_type=${CONF_TYPES} \
--configs=${CONFIGS} \
--seed=${SEED} \
--region_col_id=${REGION_COL_ID} \
--${USE_OLD_SUBS}
