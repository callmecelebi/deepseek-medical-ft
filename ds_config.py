from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "deepseek-ai/deepseek-coder-7b-base"
    max_length: int = 512
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"
    load_in_8bit: bool = True
    bnb_8bit_compute_dtype: str = "float16"
    bnb_8bit_use_double_quant: bool = True
    bnb_8bit_quant_type: str = "nf8"


@dataclass
class LoRAConfig:
    r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainingConfig:
    output_dir: str = "deepseek_medical_model"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    optim: str = "adamw_torch"
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0
    gradient_checkpointing: bool = True
    fp16: bool = True
    report_to: list = None

    def __post_init__(self):
        if self.report_to is None:
            self.report_to = ["wandb"]


@dataclass
class DataConfig:
    dataset_name: str = "medalpaca/medical_meadow_medqa"
    max_samples: Optional[int] = 2000
    train_split: float = 0.85
    validation_split: float = 0.15


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    lora: LoRAConfig = LoRAConfig()
    data: DataConfig = DataConfig()
    wandb_project: str = "deepseek-medical-finetuning"
    wandb_entity: Optional[str] = None
