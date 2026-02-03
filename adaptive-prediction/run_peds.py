# run_peds.py
import json
import os
import glob
import re
from collections import defaultdict
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils import data
from trajdata import AgentBatch, AgentType, UnifiedDataset
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.model_utils import UpdateMode
from trajectron.model.trajectron import Trajectron

# ================= 配置区域 =================

# [1] 数据路径 (你刚才下载成功的路径)
PED_DATA_DIR = "/scratch/work/zhangx29/data/eth_ucy_peds/"

# [2] 缓存路径 (统一放在这里，避免权限错误)
MY_CACHE_DIR = "/scratch/work/zhangx29/.unified_data_cache"

# [3] 模型路径 (你指定的 Zara1 模型)
MODEL_DIR = "/scratch/work/zhangx29/adaptive-prediction/experiments/pedestrians/kf_models/zara1_1mode_adaptive_tpp-20_Jan_2023_20_21_11"

# [4] 评估数据集 (必须与模型匹配，这里改为 zara1)
EVAL_DATASET = "eupeds_zara1-test"

# 其他设置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1


# ===========================================

def find_best_epoch(model_dir):
    """
    自动查找模型文件夹里数字最大的 .pt 文件
    """
    files = glob.glob(os.path.join(model_dir, "model_registrar-*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    # 提取数字并排序: model_registrar-100.pt -> 100
    epochs = []
    for f in files:
        match = re.search(r"model_registrar-(\d+).pt", f)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        raise FileNotFoundError("Could not parse epoch numbers from files.")

    best_epoch = max(epochs)
    print(f"Found checkpoints: {epochs}. Loading best epoch: {best_epoch}")
    return best_epoch


def load_model(model_dir: str, device: str):
    # 自动查找最佳 Epoch
    epoch = find_best_epoch(model_dir)
    save_path = Path(model_dir) / f"model_registrar-{epoch}.pt"

    print(f"Loading weights from: {save_path}")

    model_registrar = ModelRegistrar(model_dir, device)

    # 加载 config 并覆盖缓存路径
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file missing: {config_path}")

    with open(config_path, "r") as config_json:
        hyperparams = json.load(config_json)

    # [关键] 强制覆盖缓存路径，防止 Permission denied
    hyperparams["trajdata_cache_dir"] = MY_CACHE_DIR

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams


def run():
    print(f"--- Running Pedestrian Inference on {EVAL_DATASET} ---")
    print(f"Data Dir: {PED_DATA_DIR}")
    print(f"Cache Dir: {MY_CACHE_DIR}")

    # 1. 加载模型
    model, hyperparams = load_model(MODEL_DIR, DEVICE)
    if model is None: return

    # 2. 准备数据
    print("Loading Data (This triggers preprocessing)...")

    # 行人的注意力半径 (通常比车小)
    attention_radius = defaultdict(lambda: 3.0)
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 3.0

    # 配置所有可能的子数据集路径
    # [修改后] 所有数据集都指向同一个【父目录】
    # trajdata 会根据键名（如 eupeds_zara1）自动去父目录下找名为 zara1 的子文件夹
    # [修改后]
    # 既然文件夹改名了，我们只需要指向它们的【父目录】
    # trajdata 会自动在里面找到 ucy_zara01

    # [修改后] 全部指向父目录，让 trajdata 自动去匹配 ucy_zara01
    data_dirs_config = {
        "eupeds_eth": PED_DATA_DIR,
        "eupeds_hotel": PED_DATA_DIR,
        "eupeds_univ": PED_DATA_DIR,
        "eupeds_zara1": PED_DATA_DIR,  # 它会自动在 PED_DATA_DIR 下找 ucy_zara01
        "eupeds_zara2": PED_DATA_DIR,
    }

    dataset = UnifiedDataset(
        desired_data=[EVAL_DATASET],
        history_sec=(0.1, 3.2),  # ETH/UCY 标准设置: 8帧历史
        future_sec=(4.8, 4.8),  # ETH/UCY 标准设置: 12帧未来
        agent_interaction_distances=attention_radius,
        incl_robot_future=False,
        incl_raster_map=False,  # 行人不需要地图
        only_predict=[AgentType.PEDESTRIAN],  # 只预测行人
        no_types=[AgentType.VEHICLE],  # 忽略车辆
        num_workers=0,
        cache_location=MY_CACHE_DIR,
        data_dirs=data_dirs_config,
        verbose=True,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(pad_format="right"),
        num_workers=0
    )

    # 3. 开始推理循环
    print(f"Starting Inference on {len(dataset)} pedestrians...")
    model.reset_adaptive_info()
    results = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # 运行自适应预测
            model.adaptive_predict(batch, update_mode=UpdateMode.ITERATIVE)

            # 评估
            eval_results = model.predict_and_evaluate_batch(batch)

            for agent_type, metric_dict in eval_results.items():
                for metric, value in metric_dict.items():
                    results[f"{agent_type}_{metric}"].append(value.item())

    # 4. 打印摘要并保存
    print("\n--- Inference Complete ---")
    print("Average Metrics:")
    for key, val_list in results.items():
        print(f"{key}: {np.mean(val_list):.4f}")

    os.makedirs("results_peds", exist_ok=True)
    df = pd.DataFrame.from_dict(results)
    save_path = f"results_peds/{EVAL_DATASET}_results.csv"
    df.to_csv(save_path, index=False)
    print(f"Successfully saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    run()