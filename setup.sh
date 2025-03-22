#!/bin/bash

# Gerekli dizinleri oluştur
mkdir -p deepseek_medical_model
mkdir -p logs

# Python virtual environment oluştur
python3 -m venv venv
source venv/bin/activate

# Gerekli paketleri yükle
pip install --upgrade pip
pip install -r requirements.txt

# CUDA ayarlarını kontrol et
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python3 -c "import torch; print(f'Current CUDA device: {torch.cuda.current_device()}')"
python3 -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

# Weights & Biases login (opsiyonel)
# wandb login

echo "Kurulum tamamlandı. Eğitimi başlatmak için:"
echo "source venv/bin/activate"
echo "python ds_finetune.py" 