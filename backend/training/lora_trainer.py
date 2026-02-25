"""
LoRA Trainer Module
Main training logic for fine-tuning models with LoRA adapters.
"""

import os
import torch
from typing import Optional, Union, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer

from .model_config import LoRAConfig as CustomLoRAConfig
from .data_processor import TrainingDataProcessor

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA Fine-tuning Trainer
    Handles model loading, training, and adapter saving.
    """
    
    def __init__(self, config: Optional[CustomLoRAConfig] = None):
        """
        Initialize LoRA trainer
        
        Args:
            config: LoRA configuration (uses default if None)
        """
        self.config = config or CustomLoRAConfig()
        self.model = None
        self.tokenizer = None
        self.data_processor = TrainingDataProcessor()
        
        # Setup logging
        self._setup_logging()
        
        # Check CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("CUDA not available! Training on CPU will be very slow.")
        else:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config.logging_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config.base_model_name}")
        
        # Quantization config
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.use_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True,
            add_bos_token=True,
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )
        
        # Prepare model for k-bit training
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing
            )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model"""
        logger.info("Setting up LoRA adapters...")
        
        # LoRA config
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias=self.config.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA adapters applied successfully")
    
    def prepare_training_args(self) -> TrainingArguments:
        """Prepare training arguments"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy=self.config.evaluation_strategy if self.config.do_eval else "no",
            eval_steps=self.config.eval_steps if self.config.do_eval else None,
            max_steps=self.config.max_steps,
            report_to=["tensorboard"],
            save_strategy="steps",
            load_best_model_at_end=self.config.do_eval,
        )
        
        return training_args
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
    ) -> Dict[str, Any]:
        """
        Run LoRA fine-tuning
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        
        Returns:
            Training metrics
        """
        logger.info("Starting LoRA fine-tuning...")
        
        # Ensure model and tokenizer are loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
            self.setup_lora()
        
        # Prepare training arguments
        training_args = self.prepare_training_args()
        
        # Create trainer using SFTTrainer for instruction tuning
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
        )
        
        # Train
        logger.info("Training started...")
        train_result = trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {self.config.output_dir}")
        
        return metrics
    
    def train_from_json(
        self,
        training_data_path: Union[str, Path],
        train_test_split: float = 0.1
    ) -> Dict[str, Any]:
        """
        Train model from JSON training data file
        
        Args:
            training_data_path: Path to training JSON file
            train_test_split: Fraction for test split
        
        Returns:
            Training metrics
        """
        logger.info(f"Loading training data from: {training_data_path}")
        
        # Create dataset
        dataset_dict = self.data_processor.create_dataset(
            training_data_path,
            train_test_split=train_test_split
        )
        
        # Validate dataset
        stats = self.data_processor.validate_dataset(dataset_dict['train'])
        logger.info(f"Dataset statistics: {stats}")
        
        # Train
        eval_dataset = dataset_dict.get('test', None) if self.config.do_eval else None
        metrics = self.train(
            train_dataset=dataset_dict['train'],
            eval_dataset=eval_dataset
        )
        
        return metrics
    
    @staticmethod
    def load_trained_model(
        model_path: Union[str, Path],
        base_model_name: Optional[str] = None,
        device_map: str = "auto"
    ):
        """
        Load a trained LoRA model for inference
        
        Args:
            model_path: Path to saved LoRA adapters
            base_model_name: Base model name (will try to infer if None)
            device_map: Device mapping strategy
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = Path(model_path)
        
        # Try to load config to get base model name
        if base_model_name is None:
            config_path = model_path / "adapter_config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path")
        
        if base_model_name is None:
            raise ValueError("Could not determine base model name. Please provide it explicitly.")
        
        logger.info(f"Loading base model: {base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load LoRA adapters
        logger.info(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model, str(model_path))
        
        logger.info("Model loaded successfully")
        return model, tokenizer


def quick_train(
    training_data_path: Union[str, Path],
    output_dir: Union[str, Path] = "./models/lora_query_optimizer",
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    epochs: int = 3,
    use_lightweight: bool = True
) -> Dict[str, Any]:
    """
    Quick training function with sensible defaults
    
    Args:
        training_data_path: Path to training JSON file
        output_dir: Where to save the trained model
        model_name: Base model to fine-tune
        epochs: Number of training epochs
        use_lightweight: Use lightweight config for lower memory usage
    
    Returns:
        Training metrics
    """
    from .model_config import get_lightweight_config, get_default_config
    
    # Get config
    config = get_lightweight_config() if use_lightweight else get_default_config()
    config.base_model_name = model_name
    config.output_dir = str(output_dir)
    config.num_train_epochs = epochs
    
    # Create trainer and train
    trainer = LoRATrainer(config)
    metrics = trainer.train_from_json(training_data_path)
    
    return metrics
