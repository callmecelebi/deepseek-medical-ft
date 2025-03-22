import os
import torch
import time
import psutil
import gc
import sys
import numpy as np
import warnings
import logging
from tqdm.auto import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from sklearn.metrics import accuracy_score, f1_score
from ds_config import Config

# Uyarıları ve log seviyelerini ayarla
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for logger_name in ["transformers", "datasets", "accelerate", "huggingface_hub"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def log_memory_usage():
    """GPU ve CPU bellek kullanımını göster"""
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(
            f"GPU Bellek: {gpu_mem_alloc:.2f} GB (Rezerve: {gpu_mem_reserved:.2f} GB)",
            flush=True,
        )

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3
    print(f"CPU Bellek: {cpu_mem:.2f} GB", flush=True)


def clear_memory():
    """Belleği temizle"""
    torch.cuda.empty_cache()
    gc.collect()
    print("Bellek temizlendi", flush=True)


def prepare_dataset(examples, tokenizer, max_length=64):
    """Veri setini model için hazırla - Düşük bellek versiyonu"""
    texts = []
    for i in range(len(examples["question"])):
        # Sadece soru ve cevabı içer - açıklamayı dahil etme (bellek tasarrufu için)
        options = examples["options"][i]
        options_text = (
            "\n".join([f"{k}: {v}" for k, v in options[0].items()])
            if isinstance(options, list)
            else str(options)
        )

        # Kısa prompt
        text = f"""Soru: {examples['question'][i]}\nSeçenekler: {options_text}\nCevap: {examples['answer'][i]}"""
        texts.append(text)

    # Tokenize - düşük bellek modunda
    if texts:
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,  # Daha kısa maksimum uzunluk
            padding="max_length",
            return_tensors="pt",
        )
        return tokenized
    else:
        return {"input_ids": [], "attention_mask": []}


def load_and_prepare_data(config, tokenizer):
    """Veri setini yükle ve hazırla - Düşük bellek odaklı"""
    try:
        print("\nVeri seti yükleniyor...", flush=True)

        # Daha küçük veri seti yükle
        dataset = load_dataset(
            "json",
            data_files=config.dataset_path,
            split="train[:600]",  # Sadece ilk 600 örneği al
        )

        # Bellek temizle
        clear_memory()

        # Küçük alt kümeler oluştur
        shuffled_dataset = dataset.shuffle(seed=42)
        train_size = 500  # Daha küçük eğitim seti
        val_size = 60  # Daha küçük validasyon seti

        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))

        print(f"Toplam veri seti: {len(dataset)} örnek")
        print(f"Eğitim seti: {len(train_dataset)} örnek")
        print(f"Validasyon seti: {len(val_dataset)} örnek")

        # İşlem için bellek temizleme
        clear_memory()

        # Tokenizasyon işlemi - daha kısa sekans uzunluğu
        def tokenize_function(examples):
            texts = []
            for q, o, a in zip(
                examples["question"], examples["options"], examples["answer"]
            ):
                # Seçenekleri formatlama - basit ve kısa format
                options_text = (
                    "\n".join([f"{k}: {v}" for k, v in o[0].items()])
                    if isinstance(o, list)
                    else str(o)
                )
                # Ultra kısa prompt
                text = f"Q: {q} Options: {options_text} A: {a}"
                texts.append(text)

            return tokenizer(
                texts,
                truncation=True,
                max_length=64,  # Daha kısa max_length - 64 token
                padding="max_length",
                return_tensors="pt",
            )

        # Daha küçük batch'lerle veri setlerini işleme, bellek dostu
        print("Veri seti tokenize ediliyor...", flush=True)

        # Eğitim verisi
        train_data = train_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,  # Daha küçük batch size
            num_proc=1,
            remove_columns=train_dataset.column_names,
            desc="Eğitim verileri",
        )

        # İşlemi bitir, belleği temizle
        clear_memory()

        # Validation verisi
        val_data = val_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=8,  # Daha küçük batch size
            num_proc=1,
            remove_columns=val_dataset.column_names,
            desc="Validasyon verileri",
        )

        print(f"Eğitim veri seti hazır: {len(train_data)} örnek")
        print(f"Validasyon veri seti hazır: {len(val_data)} örnek")

        # Son bir kez bellek temizleme
        clear_memory()

        return train_data, val_data

    except Exception as e:
        print(f"Veri hazırlama hatası: {str(e)}", flush=True)
        import traceback

        print(f"Hata detayı:\n{traceback.format_exc()}", flush=True)
        raise


def compute_metrics(eval_preds):
    """Değerlendirme metriklerini hesapla"""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Padding tokenlerini maskele
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def main():
    try:
        baslangic_zamani = time.time()
        print("Program başlatılıyor...", flush=True)

        # Konfigürasyon
        config = Config()
        print("Konfigürasyon yüklendi", flush=True)
        log_memory_usage()

        # Ortam değişkenleri
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        print("Ortam değişkenleri ayarlandı", flush=True)

        # WandB başlat
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
        )

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name, trust_remote_code=config.model.trust_remote_code
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Veri setlerini hazırla
        train_data, val_data = load_and_prepare_data(config, tokenizer)
        print("Veri hazırlığı tamamlandı", flush=True)
        log_memory_usage()
        clear_memory()

        # 4-bit Quantization config - Maksimum GPU bellek optimizasyonu
        print("Model yükleniyor...", flush=True)

        # GPU'yu tamamen temizle
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        print("Mevcut GPU bellek kullanımı:", flush=True)
        log_memory_usage()

        # Sistem belleğinin çoğunu serbest bırakmak için
        print("Sistem belleğini boşaltmaya çalışıyor...", flush=True)
        try:
            os.system("sync && echo 3 > /proc/sys/vm/drop_caches")
        except:
            pass

        # Sadece GPU kullanımı için - bellek optimizasyonları
        print("GPU-only model stratejisi uygulanıyor", flush=True)

        # 8-bit quantization (daha düşük bellek gereksinimi)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        # Tüm modeli sadece GPU'ya yerleştir
        print("Model sadece GPU'ya yükleniyor...", flush=True)
        model_loading_bar = tqdm(total=1, desc="Model yükleniyor", unit="model")

        try:
            # SADECE GPU'ya yükleme deneyin
            model = AutoModelForCausalLM.from_pretrained(
                config.model.model_name,
                device_map="cuda:0",  # Sadece CUDA:0 cihazına yükle
                max_memory={0: "22GiB"},  # Tüm GPU belleğini kullan
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=False,
                attn_implementation="flash_attention_2",  # Daha düşük bellek kullanımı için Flash Attention 2 kullan
            )
            model_loading_bar.update(1)
            model_loading_bar.close()
            print("Model tamamen GPU'ya yüklendi", flush=True)
        except Exception as e:
            model_loading_bar.close()
            print(f"Hata: {e}", flush=True)
            print(
                "Model yüklenemedi. Daha fazla bellek optimizasyonu deneniyor...",
                flush=True,
            )

            # Daha agresif bellek optimizasyonları
            torch.cuda.empty_cache()
            gc.collect()

            # Daha küçük bir model deneyin (7B gibi)
            if "16b" in config.model.model_name.lower():
                smaller_model = config.model.model_name.replace("16b", "7b")
                print(f"Daha küçük model deneniyor: {smaller_model}", flush=True)

                model = AutoModelForCausalLM.from_pretrained(
                    smaller_model,
                    device_map="cuda:0",
                    max_memory={0: "22GiB"},
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_cache=False,
                )
                print("Daha küçük model GPU'ya yüklendi", flush=True)
            else:
                print(
                    "GPU belleği yeterli değil, lütfen CPU offload kullanın veya daha küçük bir model deneyin",
                    flush=True,
                )
                raise ValueError(
                    "GPU belleği yetersiz, CPU offload olmadan devam edilemiyor"
                )

        # Bellek durumunu kontrol et
        log_memory_usage()

        # Gradient checkpointing
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing etkinleştirildi", flush=True)
        clear_memory()

        # GPU'da kal - Bellek kullanımını azalt
        print("Bellek durumu (LoRA öncesi):", flush=True)
        log_memory_usage()

        # LoRA için hazırla - Sadece GPU kullanarak düşük bellek seçenekleri
        print("LoRA hazırlanıyor (GPU-only)...", flush=True)

        # Önce belleği boşalt
        clear_memory()

        try:
            # LoRA için model hazırlığı
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

            # LoRA konfigürasyonu - Sadece GPU odaklı, minimum bellek kullanımı
            # Çok az modülü hedefle, bu şekilde bellek kullanımı azalır
            target_modules = ["q_proj"]  # Sadece q_proj projeksiyonları
            lora_config = LoraConfig(
                r=1,  # LoRA rank - minimum
                lora_alpha=8,  # LoRA alpha
                lora_dropout=0.0,  # Dropout devre dışı
                bias="none",  # Bias öğrenme devre dışı
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                modules_to_save=None,  # Hiçbir modülü tamamen kaydetme
            )

            # LoRA'yı uygula ve parametreleri göster
            print("LoRA modeli oluşturuluyor...", flush=True)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"LoRA hazırlama hatası: {str(e)}", flush=True)
            print("Alternatif LoRA yapılandırması deneniyor...", flush=True)

            # Daha basit, daha az modül hedefleyen alternatif
            try:
                model = prepare_model_for_kbit_training(model)
                # Daha minimal bir LoRA yapılandırması
                lora_config = LoraConfig(
                    r=1,
                    lora_alpha=4,
                    target_modules=["q_proj"],  # Sadece q_proj (daha az bellek)
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                model = get_peft_model(model, lora_config)
                print("LoRA modeli alternatif yapılandırma ile oluşturuldu", flush=True)
            except Exception as e2:
                print(f"Alternatif LoRA hazırlama hatası: {str(e2)}", flush=True)
                raise

        clear_memory()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Eğitim için bellek durumunu kontrol et
        print("Bellek durumu (Eğitim öncesi):", flush=True)
        log_memory_usage()

        # Eğitim argümanları - Tamamen GPU odaklı
        training_args = TrainingArguments(
            output_dir=config.training.output_dir,
            num_train_epochs=1.0,  # Tek epoch
            per_device_train_batch_size=1,  # Minimum batch size
            per_device_eval_batch_size=1,  # Minimum eval batch size
            gradient_accumulation_steps=64,  # Daha fazla accumulation - bellek dostu
            learning_rate=2e-4,  # Öğrenme oranı
            weight_decay=0.0,  # Weight decay devre dışı - daha az bellek
            warmup_ratio=0.0,  # Warmup devre dışı - daha az bellek
            logging_steps=10,
            save_strategy="no",  # Ara kayıt yok - bellek tasarrufu
            evaluation_strategy="steps",
            eval_steps=25,  # Daha seyrek değerlendirme
            load_best_model_at_end=False,  # Performans için tasarruf
            fp16=True,  # FP16 aktif
            fp16_full_eval=True,  # FP16 ile eval
            fp16_opt_level="O3",  # En agresif bellek optimizasyonu
            report_to=["wandb"],
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            max_grad_norm=0.3,  # Gradient clipping
            dataloader_num_workers=0,  # İşçi yok
            remove_unused_columns=False,
            optim="adamw_torch_fused",  # GPU için optimize edilmiş optimizer
            dataloader_pin_memory=False,  # Pin memory devre dışı
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Eğitim süresi hesapla
        hazirlik_suresi = time.time() - baslangic_zamani
        print(f"Hazırlık tamamlandı. Süre: {hazirlik_suresi:.2f} saniye", flush=True)
        log_memory_usage()

        # Eğitim başlat
        print("Eğitim başlıyor...", flush=True)
        egitim_baslangic = time.time()
        trainer.train()
        egitim_suresi = time.time() - egitim_baslangic

        # Eğitim sonrası
        print(f"Eğitim tamamlandı. Süre: {egitim_suresi:.2f} saniye", flush=True)
        trainer.save_model()
        print("Model kaydedildi", flush=True)

        # WandB kapat
        wandb.finish()

        # Toplam süre
        toplam_sure = time.time() - baslangic_zamani
        print(f"İşlem tamamlandı. Toplam süre: {toplam_sure:.2f} saniye", flush=True)

    except Exception as e:
        print(f"\nAna programda hata oluştu: {str(e)}", flush=True)
        print(f"Hata tipi: {type(e).__name__}", flush=True)
        import traceback

        print(f"Hata detayı:\n{traceback.format_exc()}", flush=True)

    finally:
        # Hata olsa da olmasa da WandB'yi kapat
        try:
            wandb.finish()
        except:
            pass

        print(f"\nProgram sonlandı")


if __name__ == "__main__":
    main()
