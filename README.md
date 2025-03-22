# DeepSeek Medical Model Fine-tuning

Bu proje, DeepSeek-7B modelini tıbbi soru-cevap veri seti üzerinde fine-tuning yapmak için kullanılır.

## Kurulum

1. Gerekli paketleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Weights & Biases hesabınızı yapılandırın (opsiyonel):

```bash
wandb login
```

## Kullanım

1. Konfigürasyonu düzenleyin:

   - `ds_config.py` dosyasında model ve eğitim parametrelerini ayarlayın
   - GPU bellek kullanımına göre batch size ve gradient accumulation steps'i ayarlayın

2. Eğitimi başlatın:

```bash
python ds_finetune.py
```

## Önemli Parametreler

- `model_name`: "deepseek-ai/deepseek-coder-7b-base"
- `max_length`: 512
- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 4
- `learning_rate`: 2e-4
- `num_train_epochs`: 5

## Çıktılar

- Eğitilmiş model: `deepseek_medical_model/` dizininde saklanır
- Eğitim logları: Weights & Biases üzerinde görüntülenebilir (opsiyonel)

## Sistem Gereksinimleri

- Python 3.8+
- CUDA destekli GPU (en az 48GB VRAM önerilir)
- En az 32GB RAM
- En az 100GB disk alanı

## Notlar

- Model 7B parametre içerir
- A6000 GPU için optimize edilmiştir
- LoRA fine-tuning kullanır
- Gradient checkpointing aktif
- FP16 precision kullanır
