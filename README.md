# CONFOCAL — Conformal outage-controlled near-field beam tracking with range-limited delay–phase focusing

Reference implementation of the simulator, the proposed tracker (CONFOCAL), the five baselines,
the ablations and every experiment and figure reported in the accompanying manuscript.

Everything is plain Python with NumPy/SciPy/Matplotlib; no GPU is needed.

## Files (single flat folder)

| File | Role |
|---|---|
| `config.py` | System, hardware, frame and scenario parameters (Table II of the paper) |
| `channel.py` | Wideband spherical-wave channel, blockers (visibility regions), scatterers |
| `hardware.py` | Delay–phase beam synthesis: ideal, standard (clipped) and offset-projected TTDs; power model |
| `mobility.py` | Trajectory generator with OU turn-rate/acceleration and Poisson manoeuvres (regimes A, P, B) |
| `estimator.py` | Polar probe grid, parabolic refinement, Doppler range rate, group-wise visibility probing |
| `predictor.py` | Heteroscedastic multi-step MLP forecaster (NumPy, Adam) and Cartesian EKF baseline |
| `controller.py` | Region-robust hardware-aware DPP optimiser and online conformal (ACI) outage control |
| `sim.py` | Closed-loop frame simulator; definitions of all methods and ablations |
| `train_predictor.py` | Builds the training set (seeds disjoint from test) and trains the forecaster |
| `tune_ekf.py` | Selects the EKF process noise by one-step polar NLL |
| `run_experiments.py` | Main, ablation and sensitivity job grids (resumable, one pickle per job) |
| `analyze.py` | Per-trajectory aggregation, bootstrap CIs, Wilcoxon + Holm, rank-biserial effect sizes |
| `fig_arch.py`, `fig_mech.py`, `fig_results.py`, `fig_example.py` | All manuscript figures |
| `run_all.sh` | End-to-end reproduction |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh            # trains, runs 936 closed-loop jobs, analyses, draws figures
```

`run_experiments.py <main|abl|sens> <n_processes>` can be run stage by stage; finished jobs are skipped.
Outputs go to `results/` (job pickles, `summary.json`) and `figures/`.
On two CPU cores the full grid takes roughly 3–4 hours.

## Seeds and leakage control

* Forecaster training trajectories: seed 100; EKF tuning: seed 200.
* Test trajectory *i*: environment and trajectory seed `10000+i`, measurement-noise seed `20000+i`
  (regime A: i = 0–29, regime P: 500–511, regime B: 900–911).
* All methods see identical environments, trajectories and noise seeds (paired design).

## Notes

* The channel and hardware are simulated; there is no measured dataset.
* `hardware.beam(mode='aware')` implements the offset projection (exact band-average criterion; Lemma 1 gives its
  small-residual form); `controller.optimise` implements the region-robust alternating projected optimisation.
* `controller.OnlineConformal` implements the horizon-pooled normalised scores and the per-horizon adaptive levels
  driven by the outage indicator (Proposition 1).
* `predictor.EKFTracker` is the likelihood-tuned EKF baseline; `predictor.MLPForecaster` the heteroscedastic forecaster.

## License

MIT
