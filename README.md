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
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 8
- `learning_rate`: 2e-4
- `num_train_epochs`: 5
- `load_in_8bit`: True
- `lora_r`: 4
- `lora_alpha`: 8

## Çıktılar

- Eğitilmiş model: `deepseek_medical_model/` dizininde saklanır
- Eğitim logları: Weights & Biases üzerinde görüntülenebilir (opsiyonel)

## Sistem Gereksinimleri

- Python 3.8+
- CUDA destekli GPU (NVIDIA A5000 24GB VRAM)
- En az 32GB RAM
- En az 100GB disk alanı

## Optimizasyonlar

- 8-bit quantization kullanılıyor
- Düşük batch size (2) ve yüksek gradient accumulation (8)
- LoRA fine-tuning (r=4, alpha=8)
- Gradient checkpointing aktif
- FP16 precision
- Düşük CPU bellek kullanımı

## Veri Seti

Veri seti formatı için `example_dataset.json` dosyasına bakınız. Veri seti şu yapıda olmalıdır:

- Train ve validation split'leri
- Her örnek için:
  - Soru
  - Seçenekler
  - Doğru cevap
  - Açıklama
