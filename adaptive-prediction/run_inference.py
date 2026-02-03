# run_mini_inference.py
import json
import os
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
import trajectron.visualization as visualization
import trajdata.visualization.vis as trajdata_vis
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# [CRITICAL] Pointing to your specific paths
NUSC_DATA_DIR = "/scratch/work/zhangx29/data/nuScenes"
CACHE_DIR = "/scratch/work/zhangx29/.unified_data_cache"

# [CRITICAL] Using the MINI dataset name
EVAL_DATASET = "nusc_mini-mini_val"

# Model paths (Using the pre-trained ones included in the repo)
# We assume you are running this from the 'adaptive-prediction' root folder
BASE_MODEL_DIR = "experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45"
ADAPTIVE_MODEL_DIR = "experiments/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01"

HISTORY_SEC = 2.0
PREDICTION_SEC = 6.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Keep small for safety


def load_model(model_dir, device, epoch=20):
    save_path = Path(model_dir) / f"model_registrar-{epoch}.pt"

    if not os.path.exists(save_path):
        print(f"Error: Model file not found at {save_path}")
        return None, None

    model_registrar = ModelRegistrar(model_dir, device)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    # OVERRIDE: Force the model to use YOUR cache directory
    hyperparams["trajdata_cache_dir"] = CACHE_DIR

    trajectron = Trajectron(model_registrar, hyperparams, None, device)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    checkpoint = torch.load(save_path, map_location=device)
    trajectron.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return trajectron, hyperparams


def run():
    print(f"--- Initializing Evaluation on {EVAL_DATASET} ---")
    print(f"Data Directory: {NUSC_DATA_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")

    # 1. Load Model
    print("Loading Adaptive Model...")
    model, hyperparams = load_model(ADAPTIVE_MODEL_DIR, DEVICE)
    if model is None: return

    # 2. Prepare Dataset
    print("Loading Data (This triggers preprocessing)...")

    attention_radius = defaultdict(lambda: 20.0)
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
    attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0

    map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

    # [CRITICAL FIX] We define the dictionary key as 'nusc_mini' to match the dataset
    dataset = UnifiedDataset(
        desired_data=[EVAL_DATASET],
        history_sec=(0.1, HISTORY_SEC),
        future_sec=(PREDICTION_SEC, PREDICTION_SEC),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        raster_map_params=map_params,
        only_predict=[AgentType.VEHICLE],
        no_types=[AgentType.UNKNOWN],
        num_workers=0,
        cache_location=CACHE_DIR,
        data_dirs={
            "nusc_mini": NUSC_DATA_DIR,  # Correct key for mini dataset
        },
        verbose=True,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(pad_format="right"),
        num_workers=0
    )

    # 3. Run Inference Loop
    print(f"Starting Inference on {len(dataset)} agents...")

    model.reset_adaptive_info()

    results = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Run prediction
            # For mini dataset, we might not have enough history for adaptive methods to fully warm up
            # but we run it in standard mode here.
            model.adaptive_predict(batch, update_mode=UpdateMode.ITERATIVE)

            # Evaluate
            eval_results = model.predict_and_evaluate_batch(batch)

            for agent_type, metric_dict in eval_results.items():
                for metric, value in metric_dict.items():
                    results[f"{agent_type}_{metric}"].append(value.item())

            # # Limit to first 50 samples for quick testing
            # if i >= 50:
            #     break

    # 4. Print Summary
    print("\n--- Inference Complete ---")
    print("Average Metrics (First 50 samples):")
    for key, val_list in results.items():
        print(f"{key}: {np.mean(val_list):.4f}")

        # 4. 打印摘要
        print("\n--- Inference Complete ---")
        print("Average Metrics:")
        for key, val_list in results.items():
            print(f"{key}: {np.mean(val_list):.4f}")

        # --- [新增] 5. 保存结果到文件 ---
        print("Saving results to disk...")

        # 确保 results 文件夹存在
        os.makedirs("results", exist_ok=True)

        # 将字典转换为 DataFrame 并保存
        df = pd.DataFrame.from_dict(results)
        save_path = f"results/{EVAL_DATASET}_results.csv"
        df.to_csv(save_path, index=False)

        print(f"Successfully saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    run()