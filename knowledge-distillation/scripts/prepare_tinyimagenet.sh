#!/bin/bash
set -e

PROJECT_DIR="/scratch/work/zhangx29/knowledge-distillation"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
URL="http://cs231n.stanford.edu/tiny-imagenet-200.zip"
ZIP_PATH="$DATA_DIR/tiny-imagenet-200.zip"
ROOT_DIR="$DATA_DIR/tiny-imagenet-200"

mkdir -p "$DATA_DIR"

if [ ! -f "$ZIP_PATH" ]; then
  echo "Downloading Tiny-ImageNet..."
  wget -O "$ZIP_PATH" "$URL"
fi

if [ ! -d "$ROOT_DIR" ]; then
  echo "Extracting Tiny-ImageNet..."
  unzip -q "$ZIP_PATH" -d "$DATA_DIR"
fi

VAL_IMG_DIR="$ROOT_DIR/val/images"
VAL_ANNOTATIONS="$ROOT_DIR/val/val_annotations.txt"

if [ -d "$VAL_IMG_DIR" ] && [ -f "$VAL_ANNOTATIONS" ]; then
  echo "Reorganizing validation images..."
  while read -r img cls _; do
    mkdir -p "$ROOT_DIR/val/$cls"
    mv "$VAL_IMG_DIR/$img" "$ROOT_DIR/val/$cls/" 2>/dev/null || true
  done < "$VAL_ANNOTATIONS"
  rmdir "$VAL_IMG_DIR" 2>/dev/null || true
fi

echo "Tiny-ImageNet ready at: $ROOT_DIR"

