# Find the Fruit：寻找被遮挡的水果

[English](Readme.md)

本项目是基于 Isaac Lab 的 Direct RL 环境，通过 PPO 学习机械臂动作，改善指定水果在相机中的可见性。当前成功判据是近似可见性指标，没有实现水果抓取或采摘。

相关论文：[Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation](https://arxiv.org/abs/2505.16547)。保留原项目的论文引用；论文中的实机结果不代表当前本地代码已经复现这些结果。

## 红球和绿球代表什么

场景包含 MyBuddy 机械臂、可变形植物、固定 RGB-D 相机、灯光和两个模拟水果。球体在 `find_the_fruit_env.py` 的 `_setup_scene()` 中创建：

```python
fruit_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
# 使用 SphereCfg(radius=0.03)，关闭重力
```

| 对象 | 颜色 | 含义 |
|---|---|---|
| `GoalCube_0` | 红色 | 第 0 个候选水果 |
| `GoalCube_1` | 绿色 | 第 1 个候选水果 |

虽然命名为 Cube，实际是半径 3 cm 的球体。颜色只区分候选目标，不代表成功、失败或成熟度。`env_cfg.py` 中 `num_goal_cubes = 2`，因此默认不创建颜色表中的蓝球。

每次 `_reset_idx()` 重置时，为每个并行环境独立随机选择 `target_cube_idx`：0 表示红球，1 表示绿球。球体不会因被选中而改变颜色，所以仅看颜色无法判断本回合目标。重置还会随机调整球的 X 坐标及部分初始关节角度、恢复植物状态并清空可见性历史。

## 核心功能实现

1. **指定目标**：将仿真中已知的水果三维位置投影到图像，生成目标矩形掩码。这里使用仿真真值，不是水果检测网络。
2. **构造观测**：`_get_observations()` 将去均值后的 RGB、目标掩码及深度图拼接为 `(256, 256, 5)` 图像，另提供六个关节位置和末端相对机器人基座的三维位置。
3. **策略决策**：`agents/skrl_ppo_cfg.yaml` 配置 CNN 图像特征和全连接关节、末端特征，融合后用于策略与价值估计。
4. **执行动作**：`_pre_physics_step()` 将六维动作限制到 `[-1, 1]`，按 `当前关节位置 + 动作 × dt × action_scale` 生成关节位置目标；`_apply_action()` 下发目标。
5. **评价与重置**：根据遮挡、接触和动作幅度计算奖励，成功、接触或超时后结束回合。

物理仿真为 60 Hz，控制为 30 Hz（`decimation = 2`），回合时长上限为 5 秒。`scene_builder.py` 负责植物可变形物理和连接约束。

## 奖励函数在哪里

主奖励函数是 `find_the_fruit_env.py` 中的 `FindTheFruitEnv._get_rewards()`。令 `O` 表示目标掩码与深度植物掩码交集中的像素数，总奖励由以下五项相加：

| 奖励项 | 当前程序计算方式 |
|---|---|
| 可见性奖励 | `10 × (1 - O / (MASK_SIZE × MASK_SIZE))`，`MASK_SIZE = 30` |
| 可见性达标奖励 | `O <= 160` 时 `+3` |
| 持续达标奖励 | 连续 10 次可见性判断达标时 `+20` |
| 动作惩罚 | 达标时 `-0.06 × norm(clamp(action, -1, 1))` |
| 接触惩罚 | 任意被监测连杆净接触力非零时 `-5` |

`_calculate_contact_reward()` 计算接触惩罚，不区分接触的是植物还是其他物体。`_get_dones()` 将接触或已存储的成功标志作为终止条件，将超时作为截断条件。成功标志在奖励计算中设置，终止检查按环境执行顺序读取该标志。

**遮挡判定是近似值**：`vision_utils.py` 的 `calculate_plant_mask()` 将深度在 0.1～0.4 米之间的所有像素当作植物，并没有进行语义分割，因此不能把奖励解释成精确的水果可见比例。此外，当前掩码生成器在 `MASK_SIZE = 30` 时可生成内部 31×31 的区域，而奖励仍除以 900，归一化面积并不完全一致。

## 在本机运行训练

`run_train.sh` 使用本机现有安装，不执行依赖安装：

| 组件 | 路径或已测试版本 |
|---|---|
| Conda 环境 | `/home/cym/anaconda3/envs/lab2.1` |
| Isaac Lab | `/home/cym/isaac_sim_projects/IsaacLab-2.1.0` |
| Isaac Sim | `/home/cym/isaac-sim-4.5.0` |
| skrl | 2.0.0 |
| PyTorch | 2.13.0+cu130 |

`train.py` 基于 Isaac Lab 2.1 官方 skrl 训练脚本，在仿真程序启动后导入并注册本项目。`run_train.sh` 自动激活 Conda 环境、加载仿真器路径。换电脑时修改该脚本中的安装路径；项目文件夹保留 `Find_the_Fruit` 名称，因为入口按此名称导入包。

```bash
cd /home/cym/ROS/Arms/Find_the_Fruit

# 小规模检查：2 个环境、2 次 PPO 迭代，共 64 个向量化环境步
bash run_train.sh --headless --num_envs 2 --max_iterations 2

# 默认正式训练：8 个环境，400,000 个向量化环境步
bash run_train.sh --headless

# 增加并行数量
bash run_train.sh --headless --num_envs 16

# 显示仿真窗口
bash run_train.sh
```

启动脚本始终启用相机。`num_envs` 是共享同一策略的并行环境数量，不是独立训练的智能体数量。`env_cfg.py` 默认 50 个环境，启动脚本默认覆盖为 8 个，可通过参数再次覆盖。每次 PPO rollout 收集 `32 × 环境数` 个样本。可逐步测试 16、32 个环境，比较显存和每秒总采样量；尚未确定最优并行数量。

PPO 主要参数：rollout 长度 32、学习轮数 5、mini-batch 数 8、初始学习率 3e-4（KL 自适应）、折扣因子 0.99、GAE 系数 0.95。网络输入已改为 skrl 2.x 的 `OBSERVATIONS[...]`。

已在本机 RTX 4090 上完成两个环境、64 步的短训练并正常退出。该验证不代表长时间稳定性或策略已收敛。

训练输出位于 `logs/skrl/mybuddy_direct/`；Hydra 也可能创建 `outputs/`。激活 `lab2.1` 后可查看曲线：

```bash
tensorboard --logdir logs/skrl/mybuddy_direct
```

## 文件说明

| 文件 | 功能 |
|---|---|
| `__init__.py` | 注册任务 `Isaac-FindTheFruit-Direct-v0` |
| `train.py`、`run_train.sh` | 训练入口和本机启动脚本 |
| `env_cfg.py` | 机器人、相机、动作观测空间及仿真参数 |
| `find_the_fruit_env.py` | 场景、动作、观测、奖励、终止和重置 |
| `scene_builder.py` | 植物变形物理与连接约束 |
| `vision_utils.py` | 投影辅助、目标掩码和深度阈值处理 |
| `agents/skrl_ppo_cfg.yaml` | PPO 及网络配置 |
| `assets.py`、`assets/` | 运行必需的 USD 模型和贴图，应保留在仓库中 |
| `utils.py` | 深度估计及噪声工具，当前主环境流程没有调用 |

## 上传修改到 Git 仓库

当前远程地址为 `https://github.com/YanmingChang666/Find_the_Fruit.git`，当前分支为 `master`。在项目目录执行：

```bash
# 查看修改，并确认日志、缓存等已忽略
 git status --short
 git diff
 git status --short --ignored

# 暂存源码、文档及必需资源；忽略文件不会被添加
 git add .
 git diff --cached --stat
 git diff --cached

# 确认暂存内容后提交并推送
 git commit -m "Add local training launcher and bilingual documentation"
 git push -u origin master
```

推送需要远程仓库写权限以及已配置的 GitHub 身份认证。若上传到另一个仓库，先在 GitHub 创建空仓库，然后执行 `git remote set-url origin <你的仓库地址>` 再推送。若远程已有更新，应先整合远程提交，不要通过强制推送覆盖他人的修改。

`.gitignore` 已排除训练日志、Hydra 输出、模型检查点、视频、Python 缓存、虚拟环境及常见编辑器和临时文件；运行必需的 USD 模型与贴图仍纳入版本控制。

忽略规则不能自动取消跟踪已提交文件。原来已跟踪的 `agents/__pycache__/__init__.cpython-311.pyc` 已从 Git 索引移除，本地文件保留；删除记录将在下次提交生效，不会清理旧提交历史。本次文档更新没有执行 commit 或 push。

## 论文引用

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

## 演示已保存的策略

使用现有 lab2.1 环境，明确指定检查点。默认两个环境，仅推理，不更新模型权重。

```bash
bash run_play.sh --real-time --checkpoint logs/skrl/mybuddy_direct/<run>/checkpoints/agent_175000.pt

# Headless smoke test
bash run_play.sh --headless --max_steps 150 --checkpoint logs/skrl/mybuddy_direct/<run>/checkpoints/agent_175000.pt
```
