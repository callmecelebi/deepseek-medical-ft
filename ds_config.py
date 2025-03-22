from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "/mnt/7bmodel/deepseek-moe-16b-base-v1.5"
    max_length: int = 128
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = True

    # GPU-odaklı device map ayarları
    device_map: str = "auto"

    # 4-bit quantization için ayarlar (bellek tasarrufu)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # Model yapılandırma
    trust_remote_code: bool = True
    use_cache: bool = False

    # Bellek yönetimi - sadece GPU odaklı
    max_memory: dict = None
    offload_folder: str = None  # offload kullanmıyoruz - sadece GPU

    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {0: "8GiB"}  # Sadece GPU bellek limiti


@dataclass
class LoRAConfig:
    r: int = 1  # Minimum rank
    lora_alpha: int = 2  # Minimum alpha
    lora_dropout: float = 0.0  # Dropout devre dışı
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None
    inference_mode: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "v_proj",
            ]  # Sadece Q ve V projeksiyonlarına uygula


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Daha az adım
    learning_rate: float = 2e-5  # Daha yüksek öğrenme oranı
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    optim: str = "adamw_torch_fused"  # GPU için optimize edilmiş optimizer
    dataloader_pin_memory: bool = True  # GPU için pin memory aktif
    dataloader_num_workers: int = 0  # Tek işçi
    gradient_checkpointing: bool = True
    fp16: bool = True
    report_to: list = field(default_factory=lambda: ["wandb"])

    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["wandb"]


@dataclass
class DataConfig:
    dataset_name: str = "medalpaca/medical_meadow_medqa"
    max_samples: Optional[int] = 2000
    train_split: float = 0.9
    validation_split: float = 0.1


@dataclass
class WandBConfig:
    project: str = "deepseek-medical-finetuning"
    entity: str = "iecelebi89-istanbul-aydin-university"


class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.lora = LoRAConfig()
        self.data = DataConfig()
        self.wandb = WandBConfig()
        self.dataset_path = "train.json"
