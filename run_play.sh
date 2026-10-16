#!/usr/bin/env bash
# Local play launcher for the existing lab2.1 / Isaac Sim 4.5.0 installation.
set -e
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/cym/anaconda3/etc/profile.d/conda.sh
conda activate lab2.1
source /home/cym/isaac-sim-4.5.0/setup_conda_env.sh
cd "$PROJECT_DIR"
exec python play.py --task Isaac-FindTheFruit-Direct-v0 --algorithm PPO --enable_cameras --num_envs 2 "$@"
