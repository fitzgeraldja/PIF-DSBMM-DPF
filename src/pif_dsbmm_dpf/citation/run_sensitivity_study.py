import os
import sys

import model.joint_factor_model as joint
import model.multi_cause_influence as causal

# import model.network_model as nm
# import model.pmf as pmf
# import model.spf as spf
import numpy as np
from absl import app, flags
from citation.process_dataset import CitationSimulator
from sklearn.metrics import mean_squared_error as mse


def post_process_influence(X, Beta):
    for t, X_t in enumerate(X):
        total_X = X_t.sum(axis=1)
        no_X = total_X == 0
        Beta[no_X, t] = 1.0
    return Beta


def get_set_overlap(Beta_p, Beta, k=50):
    scores = np.zeros(Beta.shape[1])
    for t, (beta_pt, beta_t) in enumerate(zip(Beta_p.T, Beta.T)):
        top = np.argsort(beta_t)[-k:]
        top_p = np.argsort(beta_pt)[-k:]

        scores[t] = (
            np.intersect1d(top, top_p).shape[0] / np.union1d(top, top_p).shape[0]
        )
    return scores


def main(argv):
    datadir = FLAGS.data_dir
    outdir = FLAGS.out_dir
    os.makedirs(outdir, exist_ok=True)

    Q = FLAGS.num_components
    K = FLAGS.num_exog_components
    seed = FLAGS.seed

    ct = "both"
    noise = 50.0
    confounding = 50.0
    losses = np.zeros(10)

    for i, error in enumerate(np.arange(0.1, 1.1, step=0.1)):
        simulation_model = CitationSimulator(
            datapath=datadir,
            subnetwork_size=3000,
            num_topics=3000,
            influence_shp=0.001,
            covar_2="random",
            covar_2_num_cats=5,
            seed=seed,
            do_sensitivity=True,
            sensitivity_parameter=error,
            error_rate=0.3,
        )
        simulation_model.process_dataset()
        A = simulation_model.A

        print("Adj. size and mean:", A.shape, A.mean())
        print(
            "Working on sensitivity parameter (proportion of friends with single-cause confounding):",
            error,
        )
        sys.stdout.flush()

        Y, Y_past = simulation_model.make_multi_covariate_simulation(
            noise=noise, confounding_strength=confounding, confounding_to_use=ct
        )

        Beta = simulation_model.beta
        Z = simulation_model.au_embed_1
        Gamma = simulation_model.topic_embed_1
        Alpha = simulation_model.au_embed_2
        W = simulation_model.topic_embed_2

        Beta = post_process_influence(Y_past, Beta)

        N = Y_past.shape[0]
        M = Y_past.shape[1]

        m = causal.CausalInfluenceModel(
            n_components=K, n_exog_components=P, verbose=True, model_mode="full"
        )

        joint_model = joint.JointPoissonMF(n_components=K)
        joint_model.fit(Y_past, A)
        Z_hat_joint = joint_model.Et

        pmf_model = pmf.PoissonMF(n_components=P)
        pmf_model.fit(Y_past)
        W_hat = pmf_model.Eb.T

        m.fit(Y, A, Z_hat_joint, W_hat, Y_past)

        Beta_p = m.E_beta
        score = get_set_overlap(Beta_p, Beta)
        loss = mse(Beta, Beta_p)

        print("Overlap:", score, "MSE:", loss)
        print(f"{'*' * 60}")
        sys.stdout.flush()
        losses[i] = loss

    outfile = os.path.join(outdir, "result")
    np.savez_compressed(outfile, result=losses)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    # flags.DEFINE_string('model', 'pif', "method to use selected from one of [pif, spf, unadjusted, network_pref_only, topic_only, topic_only_oracle, network_only_oracle, no_unobs (gold standard)]")
    flags.DEFINE_string(
        "data_dir",
        "../dat/pokec/regional_subset",
        "path to Pokec profiles and network files",
    )
    flags.DEFINE_string("out_dir", "../out/", "directory to write output files to")
    # flags.DEFINE_string('variant', 'z-theta-joint', 'variant for fitting per-person substitutes, chosen from one of [z-theta-joint (joint model), z-only (community model only), z-theta-concat (MF and community model outputs concatenated), theta-only (MF only)]')
    # flags.DEFINE_string('confounding_type', 'both', 'comma-separated list of types of confounding to simulate in outcome, chosen from [homophily, exog, both]')
    # flags.DEFINE_string('configs', '50,50', 'list of confounding strength configurations to use in simulation; must be in format "[confounding strength 1],[noise strength 1]:[confounding strength 2],[noise strength 2], ..."')
    flags.DEFINE_integer(
        "num_components",
        10,
        "number of components to use to fit factor model for per-person substitutes",
    )
    flags.DEFINE_integer(
        "num_exog_components",
        10,
        "number of components to use to fit factor model for per-topic substitutes",
    )
    flags.DEFINE_integer(
        "seed", 10, "random seed passed to simulator in each experiment"
    )

    app.run(main)
