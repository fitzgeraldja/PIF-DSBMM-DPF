#!/bin/bash
BASE_DIR=/scratch/fitzgeraldj/data/caus_inf_data/
export IN_DIR=${BASE_DIR}
export NUM_COMPONENTS=16
export NUM_EXOG_COMPONENTS=8
export CONF_TYPES=homophily,exog,both
export CONFIGS=50,10:50,50:50,100
MAIN_REPO_DIR=~/Documents/main_project/post_confirmation/code/PIF-DSBMM-DPF/
export MAIN_CODE_DIR=${MAIN_REPO_DIR}/src/pif_dsbmm_dpf/

num_procs=8 # max number of runs to try at once
num_jobs="\j"  # The prompt escape for number of jobs currently running

for MODEL_ITER in dsbmm_dpf;
do
	for VAR_ITER in z-theta-joint z-only z-theta-joint-ndc;
	do
		for REGION_COL_ID_ITER in main_adm1_1hot main_ctry_1hot;
		do
			for USE_OLD_SUBS_ITER in use_old_subs nouse_old_subs;
			do
				for TRY_PRES_SUBS_ITER in try_pres_subs notry_pres_subs;
				do
					for SIM_ITER in {1..5};
					do
						while ((${num_jobs@P}>=$num_procs)); do
							wait -n
						done
						export SEED=${SIM_ITER}
						export MODEL=${MODEL_ITER}
						export VARIANT=${VAR_ITER}
						export OUT_DIR=${BASE_DIR}/results/${SIM_ITER}/
						export REGION_COL_ID=${REGION_COL_ID_ITER}
						export USE_OLD_SUBS=${USE_OLD_SUBS_ITER}
						export TRY_PRES_SUBS=${TRY_PRES_SUBS_ITER}
						${MAIN_CODE_DIR}/citation/submit_scripts/run_paper_experiment.sh &
					done
				done
			done
		done
	done
done

# spf removed
for MODEL_ITER in unadjusted no_unobs topic_only_oracle network_pref_only topic_only;
do
	for SIM_ITER in {1..5};
	do
		export SEED=${SIM_ITER}
		export MODEL=${MODEL_ITER}
		export VARIANT=main
		export OUT=${BASE_DIR}/results/${SIM_ITER}/
		export REGION_COL_ID="main_adm1_1hot"
		export USE_OLD_SUBS="use_old_subs"
		${MAIN_CODE_DIR}/citation/submit_scripts/run_paper_experiment.sh
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
