# Code: Causal Valuation and Equilibrium Uniqueness in Adaptive First-Price Auctions

This package accompanies the revised Causal Valuation and Equilibrium Uniqueness in Adaptive First-Price Auctions: A Mean-Field Analysis manuscript.

## Scope and provenance

Empirical evidence uses existing Criteo Uplift v2.1 user records and iPinYou auction logs. Generated-outcome identification experiments and the eBay semi-synthetic experiment have been removed. The separate mathematical certificate benchmark uses explicitly chosen primitives; those calculations are not empirical data. Calibrated equilibrium profit and causal-selection decompositions are conditional model calculations, not observed returns.

The raw Criteo and iPinYou datasets are not included. The supplied record-level resampling output is retained unchanged in results/ref_learning_real.json. This revision reconstructs its exact counts and rates and reproduces the calibrated model calculations from stored fitted primitives. It does not claim a new raw-data fit or new interval certification.

## Quick reproduction without raw data

From this package directory, run:

```bash
python code/audit_real_data.py --figure reference_selection_real_data.pdf --verify-model
python tests/test_kl_policy.py
```

The audit needs numpy, scipy and matplotlib. It writes results/real_reference_audit.json and results/calibrated_model_recheck.json. The figure is identical in content to the supplement figure. All 14 model comparisons (seven references, two grids, one independently rerun starting flow) reproduce the supplied results to less than 1.1e-12. The supplied four-start spreads remain separate evidence; the audit does not recheck four starts.

The Wilson intervals quantify Monte Carlo draw frequencies conditional on the fixed datasets and retained nuisance estimates. They are not intervals for performance in a new market.

## Manuscript to evidence map

| Material | Scripts and saved results |
|---|---|
| Certificate benchmark, interval uniqueness and entropy trade-off | v40_figs.py, grid_certificate.py, cert_interval.py, envelope_gap_045.py; corresponding JSON files |
| Conditional uniqueness for anchored play | kl_ref_table.py, cert_calib.py; kl_ref_table.json |
| Fixed-flow KL deviation inequalities | tests/test_kl_policy.py; finite-action numerical checks, not a proof or PDE certificate |
| Finite-population bound | finite_n_v5.py; finite_n_v5.json (constants of the coupling proof, two-sided bounds of Table 6; finite_n_v4.py evaluates the earlier, looser bound) |
| Calibrated selection decomposition | calib_full.py; calib_full.json / criteo_offsets / selection_split |
| iPinYou allocation and descriptive competition patterns | season1_day.py, calib_season1.py, ctr_bins.py, s3_summary.py; saved outputs |
| Criteo exposure estimates and fitted groups | load_criteo.py, criteo_aipw.py, criteo_regimes.py, criteo_diag.py; saved outputs |
| Robustness to the data inputs (Section 9.2) | sensitivity_inputs.py; sensitivity_inputs.json |
| Journal figures (Fig1-3.pdf, ESM_Fig1.pdf; vector, Arial-compatible font) | figs_dga.py (inputs: eps_table_arb.json, eps_tradeoff.json, envelope_gap_045.json, season1_day.json, calib_season1.json, ref_learning_real.json, and pay_won.npy from the raw 2013-03-16 logs) |
| Floor censoring in the iPinYou rival law (Section 9.2) | floor_censored.py (raw bid/imp/clk logs of 2013-03-16; censored Kaplan-Meier law, join check) -> floor_censored.json, floor_censored.npz; floor_censored_calib.py (refit zeta, Arb bounds, re-solved calibration) -> floor_censored_calib.json; floor_sensitivity.py gives the earlier worst-case bounds |
| Arb bounds across entropy weights (Table 3, Figure 2 left) | cert_eps_table.py; eps_table_arb_fresh.json (the original eps_table_arb.json is kept) |
| Arb bounds at the iPinYou-fitted allocation (Table 11), without the raw log | cert_season1_only.py; calib_season1_cert_fresh.json |
| Arb bounds on the wider entropy grid (Section 5) | cert_eps_grid.py; eps_grid_arb.json |
| Calibrated profit table, both grids | calib_anchored_grid.py; calib_anchored_grid.json; independently reproduced by audit_real_data.py |
| Real-record resampling | ref_learning_real.py; original ref_learning_real.json |
| Counts including failures, conditional/unconditional rates and figure | audit_real_data.py; real_reference_audit.json |

## Full raw-data workflow

The original scripts use a working directory containing their inputs and write JSON into that working directory. To avoid overwriting packaged results, use a separate run directory and add the absolute code directory to PYTHONPATH. Copy the packaged results into that run directory when they are inputs to later computations. Store fresh outputs separately until compared with the retained originals.

1. Obtain the exact public Criteo v2.1 file. The recorded checksum is in results/criteo_sha256.txt. Run load_criteo.py with its CSV/gzip path to create criteo.npz.
2. Run criteo_regimes.py with criteo.npz. It creates criteo_regimes.json and criteo_tauW.npy. Run the AIPW and diagnostic scripts with the inputs indicated in their source.
3. Obtain iPinYou training1st logs for 20130316; expose the decompressed bid, impression, click and conversion files as bid.txt, imp.txt, clk.txt and conv.txt in the run directory. season1_day.py now explicitly emits ipy_s1.npz (won, pay, clk), the input required by ref_learning_real.py. Clicks for losses are not observed and are never treated as measured loss outcomes; the resampling script uses clicks only among wins.
4. Run ctr_bins.py and the calibration scripts. The score-scale calibration also uses pay_won.npy from season1_day.py. For the other empirical exercises, use training3rd logs for 20131023 and the season-one leaderboard file as documented in the corresponding scripts.
5. Run ref_learning_real.py with criteo.npz, criteo_tauW.npy, ipy_s1.npz and the required calibration JSONs in the run directory.

The broader code requires Python 3.10+, numpy, scipy, pandas, scikit-learn, matplotlib and python-flint. Interval calculations require python-flint/Arb. All interval certificates were re-run with python-flint 0.9.0 and reproduce every certified/not-certified verdict in the paper; see REPRODUCE_ARB.md for the environment, commands and logs.

## Optional acceleration (numba and JAX)

Two optional packages speed up the floating-point numerics. Install them with `pip install -r requirements-fast.txt`. Without them, every script runs the original numpy code.

- **numba** compiles three inner loops of the equilibrium solver in `fast_kernels.py`: the auction scores with belief interpolation, the Gibbs (softmax) step, and the mass redistribution after an auction epoch. `verify_auction.py` and `calib_model.py` call these kernels when numba is present. One evaluation of the response map is 6 to 8 times faster. The results agree with the numpy path to about 1e-16, and Picard iteration takes the same number of steps. Set `NO_NUMBA=1` to force the numpy path.
- **JAX** gives the exact Jacobian of the discretized response map by forward-mode autodiff (`jax_phi.py`). `index_test.py` and `index_norm.py` use it in place of one-sided finite differences with step 1e-6. One Jacobian takes about 0.15 s, against about 6 s for numpy finite differences (2 s with numba). It agrees with the finite-difference Jacobian to a relative error of about 1e-7, the truncation error of the finite differences. Set `JAC=fd` to force finite differences. `results/index_test.json` and `results/index_norm.json` now hold the autodiff values. The earlier finite-difference outputs are kept as `index_test_fd.json` and `index_norm_fd.json`, and they agree to every digit shown in Table 9.
- The interval certificates (`cert_interval.py`, `cert_general.py`, `cert_calib.py`) are unchanged. They use python-flint/Arb ball arithmetic, which numba and JAX cannot replace without losing the rigorous enclosure.

`python ../tests/test_accelerated.py`, run from `code/`, checks both accelerated paths against the numpy code; its output is saved in `results/test_accelerated.txt`. Rerunning `v40_figs.py`, `monotone_test.py`, `robustness.py` (including its Arb bounds), `calib_full.py` and `calib_anchored_grid.py` with numba reproduces the saved JSON to a relative difference below 6e-15.

## Interpretation and remaining checks

- Criteo's official later-release notice corrects the original advertiser-related leakage. Adjusted/unadjusted differences alone do not establish that the later release remains confounded. 
- The reference resampling analysis retains full-sample fitted groups, propensities, score scale, prior and monetary normalization. It is conditional stability analysis, not learning from scratch or independent policy evaluation.
- The original resampling output records total failures only. Reasons cannot be reconstructed by category without the raw-data rerun. The code can fail for empty price bins, zero baseline-bin clicks, nonpositive exposures/background masses or exceptions during offset construction.
- The 500 independent Gaussian-perturbation frequencies were removed from the narrative, calibration script and saved calibration output. They are not real-record bootstrap evidence.
- The calibrated model includes an exogenous tail point above the highest endogenous bid. The benchmark certificate does not automatically transfer to this extended calibration.


## Primary dataset documentation

- Criteo release and erratum: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Criteo authors' v2.1 repository: https://github.com/criteo-research/large-scale-ITE-UM-benchmark
- iPinYou competition data: https://contest.ipinyou.com/

## Fresh interval certification ( October 2026)
See REPRODUCE_ARB.md for versions, exact commands and logs.
All interval certificates were rerun with python-flint 0.9.0: cert_interval.py with 4000 cells, cert_eps_table.py, kl_ref_table.py, cert_season1_only.py, and the Arb parts of robustness.py and calib_full.py. Every certified/not-certified verdict in the paper is reproduced. Every printed bound equals the fresh upper bound or exceeds it by at most one unit in the last printed digit, because printed bounds are rounded up. A scan over theta in [8, 24], step 0.25, at eta = 0.487 certifies no value (smallest lower bound 1.00041 at theta = 16.25), which confirms the stated threshold.
