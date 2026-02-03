#!/usr/bin/env python3
import tarfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

SRC_DIR = Path("/scratch/shareddata/dldata/nuscenes/downloads")
DEST_DIR = Path("/scratch/work/zhangx29/data/nuScenes")
DEST_DIR.mkdir(parents=True, exist_ok=True)

tgz_files = list(SRC_DIR.glob("*.tgz"))

def extract_file(tgz_file):
    subdir = DEST_DIR / tgz_file.stem
    subdir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tgz_file, "r:gz") as tar:
            for member in tar.getmembers():
                tar.extract(member, path=subdir)
        return f"{tgz_file.name} 解压完成"
    except Exception as e:
        return f"{tgz_file.name} 解压失败: {e}"

print(f"找到 {len(tgz_files)} 个 tgz 文件，开始并行解压...")

# 使用 4 个进程，你可以根据服务器核心数调整
with ProcessPoolExecutor(max_workers=4) as executor:
    for result in tqdm(executor.map(extract_file, tgz_files), total=len(tgz_files)):
        print(result)

print("所有文件解压完成！")