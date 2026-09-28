#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

CUDA_VISIBLE_DEVICES=3 python src/train.py --config-dir=configs/asv_2019/ --config-name=wavlm-simple &
# CUDA_VISIBLE_DEVICES=3 python src/train.py --config-dir=configs/asv_2019/ --config-name=wav2vec2-simple &
CUDA_VISIBLE_DEVICES=2 python src/train.py --config-dir=configs/asv_2019/ --config-name=hubert-simple &


# CUDA_VISIBLE_DEVICES=7 python src/train.py --config-dir=configs/asv_2019/ --config-name=wavlm-simple data.augmentation.enabled=True &

wait