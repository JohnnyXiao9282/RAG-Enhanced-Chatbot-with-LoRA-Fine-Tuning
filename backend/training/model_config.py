"""
LoRA Configuration Module
Defines hyperparameters and model configurations for LoRA fine-tuning.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    
    # Model Configuration
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_max_length: int = 2048
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    use_8bit: bool = False
    
    # LoRA Parameters
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha scaling
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will be set in __post_init__
    lora_bias: str = "none"  # Options: "none", "all", "lora_only"
    
    # Training Parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    
    # Optimizer Settings
    optim: str = "paged_adamw_32bit"  # Memory-efficient optimizer
    lr_scheduler_type: str = "cosine"
    
    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Output Directories
    output_dir: str = "./models/lora_query_optimizer"
    logging_dir: str = "./logs/lora_training"
    
    # Training Behavior
    fp16: bool = False
    bf16: bool = False  # Use bfloat16 if available (better for modern GPUs)
    gradient_checkpointing: bool = True
    max_steps: int = -1  # -1 means use num_train_epochs
    
    # Data Processing
    max_seq_length: int = 512
    packing: bool = False
    
    # Evaluation
    evaluation_strategy: str = "steps"
    do_eval: bool = True
    
    def __post_init__(self):
        """Set default target modules and auto-detect hardware capabilities"""
        if self.lora_target_modules is None:
            # Default target modules for Mistral/Llama architecture
            self.lora_target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        
        # Auto-detect bfloat16 support
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                self.bf16 = True
                self.fp16 = False
            else:
                self.bf16 = False
                self.fp16 = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


def get_default_config(model_name: Optional[str] = None) -> LoRAConfig:
    """
    Get default configuration with optional model override
    
    Args:
        model_name: Optional model name to override default
    
    Returns:
        LoRAConfig instance
    """
    config = LoRAConfig()
    if model_name:
        config.base_model_name = model_name
    return config


def get_lightweight_config() -> LoRAConfig:
    """
    Get configuration optimized for lower memory usage
    Suitable for GPUs with 8GB VRAM
    """
    config = LoRAConfig(
        base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit=True,
        lora_r=8,  # Reduced rank
        lora_alpha=16,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        max_seq_length=512,
        gradient_checkpointing=True,
    )
    return config


def get_performance_config() -> LoRAConfig:
    """
    Get configuration optimized for performance
    Suitable for GPUs with 24GB+ VRAM
    """
    config = LoRAConfig(
        base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
        use_4bit=False,
        use_8bit=True,
        lora_r=32,  # Higher rank for better performance
        lora_alpha=64,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        gradient_checkpointing=False,
    )
    return config


# Model registry with recommended configurations
MODEL_CONFIGS = {
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "llama-2-7b": {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "target_modules": ["q_proj", "k_proj", "v_proj", "dense"],
    },
    "qwen-7b": {
        "model_name": "Qwen/Qwen-7B-Chat",
        "target_modules": ["c_attn", "c_proj", "w1", "w2"],
    },
}


def get_model_config(model_key: str) -> LoRAConfig:
    """
    Get configuration for a specific model from registry
    
    Args:
        model_key: Key from MODEL_CONFIGS (e.g., "mistral-7b")
    
    Returns:
        LoRAConfig instance configured for the specified model
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_info = MODEL_CONFIGS[model_key]
    config = LoRAConfig(
        base_model_name=model_info["model_name"],
        lora_target_modules=model_info["target_modules"],
    )
    return config
