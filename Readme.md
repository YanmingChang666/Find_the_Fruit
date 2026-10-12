# Find the Fruit

[中文说明](Readme_cn.md)

An Isaac Lab Direct RL environment for learning arm actions that improve a selected fruit's visibility. The current task evaluates an approximate visibility signal; it does not implement fruit grasping or harvesting.

Related paper: [Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation](https://arxiv.org/abs/2505.16547), Nitesh Subedi, Hsin-Jung Yang, Devesh K. Jha, and Soumik Sarkar. The paper's real-world results are not validation results for this local checkout.

## Scene and colored spheres

The scene contains a MyBuddy robot, a deformable plant, a fixed RGB-D camera, lights, and two spherical fruit proxies. In `find_the_fruit_env.py`, `_setup_scene()` defines `fruit_colors` and creates spheres with `SphereCfg(radius=0.03)`, with gravity disabled.

| Object | Color | Meaning |
|---|---|---|
| `GoalCube_0` | Red | Candidate fruit 0 |
| `GoalCube_1` | Green | Candidate fruit 1 |

Despite their names, these objects are spheres. Colors distinguish candidates, not success/failure or ripeness. `env_cfg.py` sets `num_goal_cubes = 2`. Blue is the third palette entry but is unused by default.

At each reset, `_reset_idx()` independently samples `target_cube_idx` for each environment: 0 selects red, 1 selects green. Both spheres keep their colors, so color alone does not identify the current target. The reset also randomizes fruit X positions and part of the initial joint configuration, restores the plant, and clears the visibility history.

## Core implementation

1. Project the selected fruit's known simulation position into the camera image and generate a rectangular target mask. This is privileged simulation information, not a learned fruit detector.
2. `_get_observations()` combines mean-subtracted RGB, the target mask, and depth into a `(256, 256, 5)` image. Six joint positions and the three-dimensional end-effector position relative to the robot base are also provided.
3. The PPO configuration in `agents/skrl_ppo_cfg.yaml` defines convolutional image features and fully connected joint/end-effector features, fused for policy and value estimation.
4. `_pre_physics_step()` clamps six action values to `[-1, 1]` and converts them into joint position targets: `current_position + action * dt * action_scale`. `_apply_action()` sends these targets to the robot.
5. The environment evaluates visibility, contact, and actions to compute reward and termination.

Physics runs at 60 Hz; control runs at 30 Hz (`decimation = 2`). Episodes have a five-second limit. `scene_builder.py` configures deformable plant physics and attachments.

## Rewards and termination

The main reward implementation is `FindTheFruitEnv._get_rewards()` in `find_the_fruit_env.py`. Let `O` be the number of pixels in the intersection of the selected target mask and the depth-based plant mask.

| Component | Formula / condition |
|---|---|
| Visibility shaping | `10 * (1 - O / (MASK_SIZE * MASK_SIZE))`, with `MASK_SIZE = 30` |
| Visibility bonus | `+3` when `O <= 160` |
| Sustained visibility | `+20` when the last 10 visibility checks all pass |
| Action penalty | `-0.06 * norm(clamp(action, -1, 1))`, only when visibility passes |
| Contact penalty | `-5` when any monitored link has nonzero net contact force |

`_calculate_contact_reward()` implements contact detection. It does not identify whether the contacted object is a plant. `_get_dones()` returns contact or the stored success flag as termination, and the episode time limit as truncation. The success flag is set during reward calculation, so termination consumes that stored flag according to the environment step order.

Visibility is approximate: `vision_utils.calculate_plant_mask()` classifies every depth pixel between 0.1 and 0.4 m as plant, without semantic segmentation. Consequently, this score is not an exact visible-fruit fraction. Also, the current mask generator can produce a 31-by-31 interior mask for `MASK_SIZE = 30`, while the reward denominator is 900; the normalization is not an exact mask-area ratio.

## Run training locally

The local launcher targets the existing installation:

- Conda environment: `/home/cym/anaconda3/envs/lab2.1`
- Isaac Lab: `/home/cym/isaac_sim_projects/IsaacLab-2.1.0`
- Isaac Sim: `/home/cym/isaac-sim-4.5.0`
- Tested installed libraries: skrl 2.0.0 and PyTorch 2.13.0+cu130

`train.py` is adapted from the Isaac Lab 2.1 official skrl trainer and imports this task after starting SimulationApp. `run_train.sh` activates the environment and loads simulator paths. For another machine, edit the installation paths in `run_train.sh`; keep the package directory named `Find_the_Fruit`, which the training entry point imports.

```bash
cd /home/cym/ROS/Arms/Find_the_Fruit

# Short check: 2 environments, 2 PPO iterations = 64 vectorized steps
bash run_train.sh --headless --num_envs 2 --max_iterations 2

# Default: 8 environments, 400,000 vectorized steps
bash run_train.sh --headless

# Try more parallel environments
bash run_train.sh --headless --num_envs 16

# Show the simulation window
bash run_train.sh
```

The launcher always enables cameras. `num_envs` counts parallel copies sharing a policy, not independently trained agents. The environment config defaults to 50, while the launcher selects 8 unless overridden. With rollout length 32, each PPO rollout contains `32 * num_envs` samples. Increase to 16 or 32 only after checking memory and total sample throughput; no optimal environment count has been established.

Current PPO settings: 32 rollout steps, 5 learning epochs, 8 mini-batches, learning rate 3e-4 with KL adaptation, discount 0.99, and GAE lambda 0.95. The YAML uses skrl 2.x `OBSERVATIONS[...]` inputs.

A two-environment, 64-step training check completed and exited successfully on the local RTX 4090. This is a smoke test, not evidence of convergence or long-run stability.

Outputs are saved under `logs/skrl/mybuddy_direct/`; Hydra can also create `outputs/`. Both are ignored by Git.

```bash
# After activating lab2.1
 tensorboard --logdir logs/skrl/mybuddy_direct
```

## Files

| File | Responsibility |
|---|---|
| `__init__.py` | Registers `Isaac-FindTheFruit-Direct-v0` |
| `train.py`, `run_train.sh` | Training entry point and local launcher |
| `env_cfg.py` | Robot, camera, action/observation spaces and simulation settings |
| `find_the_fruit_env.py` | Scene, action execution, observations, rewards and resets |
| `scene_builder.py` | Deformable plant and attachments |
| `vision_utils.py` | Projection helpers, target masks and depth thresholding |
| `agents/skrl_ppo_cfg.yaml` | PPO and network configuration |
| `assets.py`, `assets/` | Required USD models and textures; keep these in Git |
| `utils.py` | Additional depth estimation/noise utilities; not used in the current main environment path |

## Upload changes to Git

The configured remote is `https://github.com/YanmingChang666/Find_the_Fruit.git`, and the current branch is `master`. Run from this project directory:

```bash
# Review changes and ensure generated data are ignored
 git status --short
 git diff
 git status --short --ignored

# Stage source, documentation and required assets; .gitignore excludes generated data
 git add .
 git diff --cached --stat
 git diff --cached

# Commit and push after reviewing
 git commit -m "Add local training launcher and bilingual documentation"
 git push -u origin master
```

Push requires write access to the remote and configured GitHub authentication. If using a different repository, create an empty remote repository and change the URL with `git remote set-url origin <your-repository-url>` before pushing. If the remote has newer commits, integrate them before retrying; do not force-push over others' work.

`.gitignore` excludes logs, Hydra outputs, checkpoints, videos, Python caches, virtual environments, and common editor/temporary files. Required USD models and textures remain included. Ignore rules do not untrack existing files: the previously tracked `agents/__pycache__/__init__.cpython-311.pyc` is removed from the Git index while its local copy is preserved. This removal takes effect in the next commit; it does not rewrite history.

## Citation

If you find this code useful in your research, please cite:

```bibtex
@article{subedi2025findthefruit,
  title   = {Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation},
  author  = {Subedi, Nitesh and Yang, Hsin-Jung and Jha, Devesh K. and Sarkar, Soumik},
  journal = {arXiv preprint arXiv:2505.16547},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.16547}
}
```