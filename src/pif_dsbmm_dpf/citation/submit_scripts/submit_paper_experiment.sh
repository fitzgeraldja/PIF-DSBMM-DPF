#!/bin/bash
BASE_DIR=/scratch/fitzgeraldj/data/caus_inf_data/
# export DIR=../dat/pokec/regional_subset/
export NUM_COMPONENTS=5
export NUM_EXOG_COMPONENTS=5
export CONF_TYPES=homophily,exog,both
export CONFIGS=50,10:50,50:50,100


# ADD
# "region_col_id" in ["main_adm1_1hot", "main_ctry_1hot"]
# and
# --["use_old_subs","nouse_old_subs"]
for MODEL_ITER in dsbmm_dpf;
do
	for VAR_ITER in z-theta-joint z-only z-theta-joint-ndc;
	do
		for REGION_COL_ID_ITER in main_adm1_1hot main_ctry_1hot;
		do
			for USE_OLD_SUBS_ITER in use_old_subs nouse_old_subs;
			do
				for SIM_ITER in {1..5};
				do
					export SEED=${SIM_ITER}
					export MODEL=${MODEL_ITER}
					export VARIANT=${VAR_ITER}
					export OUT=${BASE_DIR}/results/${SIM_ITER}/
					export REGION_COL_ID=${REGION_COL_ID_ITER}
					export USE_OLD_SUBS=${USE_OLD_SUBS_ITER}
					./citation/submit_scripts/run_paper_experiment.sh
				done
			done
		done
	done
done

# spf network_pref_only topic_only removed
for MODEL_ITER in unadjusted no_unobs topic_only_oracle;
do
	for SIM_ITER in {1..5};
	do
		export SEED=${SIM_ITER}
		export MODEL=${MODEL_ITER}
		export VARIANT=main
		export OUT=${BASE_DIR}/results/${SIM_ITER}/
		export REGION_COL_ID=main_adm1_1hot
		export USE_OLD_SUBS=use_old_subs
		./citation/submit_scripts/run_paper_experiment.sh
	done
done

# for MODEL_ITER in pif;
# do
# 	for VAR_ITER in z-only z-theta-joint;
# 	do

# 		for SIM_ITER in {1..10};
# 		do
# 			export SEED=${SIM_ITER}
# 			export MODEL=${MODEL_ITER}
# 			export VARIANT=${VAR_ITER}
# 			export OUT=${BASE_DIR}/pokec_paper_results/${SIM_ITER}/
# 			./pokec/submit_scripts/run_paper_experiment.sh
# 		done
# 	done
# done
