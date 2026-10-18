#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p results figures
python3 train_predictor.py A
python3 tune_ekf.py
python3 run_experiments.py main 2
python3 run_experiments.py abl 2
python3 run_experiments.py sens 2
python3 analyze.py
python3 fig_arch.py
python3 fig_mech.py
python3 fig_results.py
python3 fig_example.py
