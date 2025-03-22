#!/bin/bash

# Virtual environment'ı aktifleştir
source venv/bin/activate

# CUDA ayarlarını kontrol et
nvidia-smi

# Eğitimi başlat
python ds_finetune.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log 