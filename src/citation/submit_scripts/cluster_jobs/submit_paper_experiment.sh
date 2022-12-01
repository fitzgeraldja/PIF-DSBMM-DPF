#!/bin/bash
BASE_DIR=/proj/sml/projects/social-influence
export DIR=/proj/sml/projects/social-influence/pokec/regional_subset/
export NUM_COMPONENTS=5
export NUM_EXOG_COMPONENTS=5
export CONF_TYPES=homophily,exog,both
export CONFIGS=50,10:50,50:50,100
export INF_STRENGTH=0.005

for MODEL_ITER in unadjusted spf network_pref_only item_only no_unobs item_only_oracle;
do
	for SIM_ITER in {1..10};
	do
		export SEED=${SIM_ITER}
		export MODEL=${MODEL_ITER}
		export VARIANT=main
		export OUT=${BASE_DIR}/pokec_paper_results/${SIM_ITER}/
		sbatch --job-name=pokec_exp_${SIM_ITER}_${MODEL_ITER} --output=${BASE_DIR}/out/pokec_experiment_${SIM_ITER}_${MODEL_ITER}.out pokec/submit_scripts/cluster_jobs/run_paper_experiment.sh
	done
done


for MODEL_ITER in pif;
do
	for VAR_ITER in z-only z-theta-joint;
	do
		
		for SIM_ITER in {1..10};
		do
			export SEED=${SIM_ITER}
			export MODEL=${MODEL_ITER}
			export VARIANT=${VAR_ITER}
			export OUT=${BASE_DIR}/pokec_paper_results/${SIM_ITER}/
			sbatch --job-name=pokec_exp_${SIM_ITER}_${VAR_ITER} --output=${BASE_DIR}/out/pokec_experiment_${SIM_ITER}_${VAR_ITER}.out pokec/submit_scripts/cluster_jobs/run_paper_experiment.sh
		done
	done
done
