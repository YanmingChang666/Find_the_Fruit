from pathlib import Path

# 资产目录：与本文件同级的 assets/ 文件夹（运行前需确保 USD 文件确实存在于此）。
_ASSETS_DIR = Path(__file__).parent / "assets"

# 机器人（带关节的机械臂 "mybuddy"），由 env_cfg.robot_cfg 引用
ROBOT_USD = str(_ASSETS_DIR / "mybuddy_rotate.usd")
# 植物几何（被 scene_builder 转成可变形软体），由 spawn_plant 引用
PLANT_USD = str(_ASSETS_DIR / "plant_v21.usd")
# 背景（拍摄背景/场景杂物，用于增加视觉复杂度），在 _setup_scene 中加载
BACKGROUND_USD = str(_ASSETS_DIR / "background.usd")