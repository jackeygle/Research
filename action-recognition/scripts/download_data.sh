#!/bin/bash
#SBATCH --job-name=download_ucf101
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
#SBATCH --output=/scratch/work/zhangx29/action-recognition/results/download_%j.out
#SBATCH --error=/scratch/work/zhangx29/action-recognition/results/download_%j.err

# Download UCF101 Dataset
# =======================

DATA_DIR="/scratch/work/zhangx29/data"
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "Starting download at: $(date)"
echo "Downloading UCF101 dataset..."

# Download UCF101 videos
if [ ! -f "UCF101.rar" ] || [ $(stat -c%s "UCF101.rar" 2>/dev/null || echo 0) -lt 6900000000 ]; then
    wget --no-check-certificate -c https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O UCF101.rar
    echo "Download completed at: $(date)"
else
    echo "UCF101.rar already exists, skipping download"
fi

# Extract using unar (available on Triton)
echo "Extracting UCF101..."
if [ ! -d "UCF101" ] || [ $(ls UCF101 | wc -l) -lt 100 ]; then
    module load p7zip 2>/dev/null || true
    if command -v unar &> /dev/null; then
        unar -o . UCF101.rar
    elif command -v 7z &> /dev/null; then
        7z x -y UCF101.rar
    else
        echo "Trying unrar..."
        unrar x -o+ UCF101.rar
    fi
fi

echo "Done at: $(date)"
echo "Dataset location: $DATA_DIR/UCF101"
echo "Number of action classes:"
ls -1 UCF101 2>/dev/null | wc -l || echo "Extraction may still be needed"
