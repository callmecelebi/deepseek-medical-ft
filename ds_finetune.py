import os
import torch
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
import bitsandbytes as bnb
from ds_config import Config
import wandb
from tqdm.auto import tqdm
import logging
import gc
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_prompt(question: str, options: list, answer: str) -> str:
    """Soru, seçenekler ve cevabı formatlı bir prompt'a dönüştürür."""
    options_text = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
    return f"""You are a medical expert. Please answer the following medical question by selecting the most appropriate option and explaining your reasoning:

Question: {question}

Options:
{options_text}

Let's think about this step by step:
1) First, let's understand the question...
2) Then, let's analyze each option...
3) Finally, let's make our decision...

Answer: {answer}
Explanation: Based on the medical knowledge and careful analysis of the options, {answer} is the most appropriate choice because..."""


def setup_model_and_tokenizer(config: Config):
    """Model ve tokenizer'ı hazırlar."""
    logger.info("Model ve tokenizer yükleniyor...")

    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name, trust_remote_code=config.model.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Model'i yükle
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        use_cache=config.model.use_cache,
        torch_dtype=torch.float16,  # A6000'de float16 kullanabiliriz
        low_cpu_mem_usage=True,  # CPU bellek kullanımını optimize et
    )

    # Model'i kbit training için hazırla
    model = prepare_model_for_kbit_training(model)

    # LoRA konfigürasyonu
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    # LoRA modelini oluştur
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def load_and_prepare_data():
    # Veri setini yükle
    dataset = load_dataset(
        "json",
        data_files={
            "train": "medical_qa_dataset.json",
            "validation": "medical_qa_dataset.json",
        },
    )

    # Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def format_example(example):
        # Her örneği formatlayarak text alanı oluştur
        if isinstance(example, dict):
            if "question" in example and "options" in example:
                text = f"Question: {example['question']}\n\nOptions:\n"
                for opt_key, opt_value in example["options"].items():
                    text += f"{opt_key}: {opt_value}\n"
                if "answer" in example:
                    text += f"\nAnswer: {example['answer']}"
                if "explanation" in example:
                    text += f"\nExplanation: {example['explanation']}"
                return {"text": text}
            else:
                return example
        return {"text": str(example)}

    # Veri setini formatlı hale getir
    formatted_dataset = dataset.map(format_example)

    # Veri setini tokenize et
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.model.max_length,
            return_tensors="pt",
        )

    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer


def compute_metrics(eval_preds):
    """Değerlendirme metriklerini hesaplar."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Mask out padding tokens
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1": f1}


def main():
    # Weights & Biases'ı başlat
    if config.training.report_to and "wandb" in config.training.report_to:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"deepseek-medical-{config.lora.r}-{config.lora.lora_alpha}",
        )

    # Veri setini ve tokenizer'ı yükle
    dataset, tokenizer = load_and_prepare_data()

    # Quantization konfigürasyonunu oluştur
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_quant_type="nf8",
    )

    # Modeli yükle
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=config.model.low_cpu_mem_usage,
        device_map=config.model.device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Modeli k-bit eğitim için hazırla
    model = prepare_model_for_kbit_training(model)

    # LoRA konfigürasyonunu ayarla
    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.lora.target_modules,
    )

    # LoRA modelini oluştur
    model = get_peft_model(model, peft_config)

    # Eğitim argümanlarını ayarla
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        evaluation_strategy=config.training.evaluation_strategy,
        eval_steps=config.training.eval_steps,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        optim=config.training.optim,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        dataloader_num_workers=config.training.dataloader_num_workers,
        gradient_checkpointing=config.training.gradient_checkpointing,
        fp16=config.training.fp16,
        report_to=config.training.report_to,
    )

    # Data collator'ı ayarla
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer'ı oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Eğitimi başlat
    trainer.train()

    # Modeli kaydet
    trainer.save_model()

    # Weights & Biases'ı kapat
    if config.training.report_to and "wandb" in config.training.report_to:
        wandb.finish()


if __name__ == "__main__":
    # Konfigürasyonu yükle
    config = Config()

    # Eğitimi başlat
    main()
