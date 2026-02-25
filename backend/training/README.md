# LoRA Fine-Tuning for Query Optimization

This directory contains the complete LoRA fine-tuning infrastructure for optimizing queries in the RAG system.

## Overview

The system fine-tunes small language models (Mistral-7B, Llama-2-7B, etc.) using LoRA (Low-Rank Adaptation) to improve query optimization for better document retrieval. This approach:

- **Reduces Memory**: Uses 4-bit quantization to run on consumer GPUs (8GB+ VRAM)
- **Fast Training**: LoRA only trains ~1% of model parameters
- **Domain-Specific**: Learns your specific document domain and query patterns
- **Efficient Inference**: Minimal overhead during query processing

## Architecture

```
backend/training/
├── __init__.py              # Package initialization
├── model_config.py          # LoRA hyperparameters and configurations
├── data_processor.py        # Training data formatting and processing
└── lora_trainer.py          # Main training loop and model management
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Enable LoRA Training

Edit `backend/config/settings.py`:

```python
LORA_ENABLED = True  # Enable LoRA fine-tuning
LORA_USE_LIGHTWEIGHT_CONFIG = True  # For 8GB VRAM GPUs
```

### 3. Upload Documents and Train

Through the UI:

1. Upload documents via the Upload panel
2. Check "Enable model training with uploaded documents"
3. Training starts automatically after data generation

Or programmatically:

```python
from ingestion.ingestion import Ingestaion

ingestion = Ingestaion(
    parser="MistralOCR",
    chunker="groupEmbeddingAlgo",
    encoder=encoder
)

# This will generate training data AND fine-tune the model
ingestion.ingest_files(
    file_path_lists=["path/to/document.pdf"],
    save_json=True,
    train_query_opt=True  # Enable training
)
```

## Configuration Options

### Model Selection

Choose from pre-configured models in `model_config.py`:

```python
from training.model_config import get_model_config

# Available models: "mistral-7b", "llama-2-7b", "phi-2", "qwen-7b"
config = get_model_config("mistral-7b")
```

### Memory Configurations

**Lightweight (8GB VRAM)**:

```python
from training.model_config import get_lightweight_config

config = get_lightweight_config()
# Uses: 4-bit quantization, rank=8, batch_size=2
```

**Performance (24GB+ VRAM)**:

```python
from training.model_config import get_performance_config

config = get_performance_config()
# Uses: 8-bit quantization, rank=32, batch_size=8
```

### Key Hyperparameters

Edit in `backend/config/settings.py`:

```python
LORA_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_EPOCHS = 3                    # Training epochs
LORA_BATCH_SIZE = 4                # Per-device batch size
LORA_RANK = 16                     # LoRA rank (8-64)
LORA_ALPHA = 32                    # LoRA alpha scaling
LORA_MAX_SEQ_LENGTH = 512          # Max sequence length
LORA_MIN_TRAINING_EXAMPLES = 100   # Min examples to start training
```

## Training Process

### Automatic Training (Recommended)

Training happens automatically when:

1. User uploads documents with "Train Model" enabled
2. System generates 100+ training examples
3. LoRA fine-tuning starts in background

### Manual Training

```python
from training.lora_trainer import quick_train

# Train from existing training data
metrics = quick_train(
    training_data_path="./data/training_data/training_data_20260225_120000.json",
    output_dir="./models/lora_query_optimizer",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    epochs=3,
    use_lightweight=True
)

print(f"Training completed! Metrics: {metrics}")
```

### Advanced Training

```python
from training.lora_trainer import LoRATrainer
from training.model_config import LoRAConfig

# Custom configuration
config = LoRAConfig(
    base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    lora_r=32,
    lora_alpha=64,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    output_dir="./models/custom_optimizer"
)

# Create trainer
trainer = LoRATrainer(config)

# Train from JSON file
metrics = trainer.train_from_json(
    training_data_path="./data/training_data/my_training_data.json",
    train_test_split=0.1  # 10% for validation
)
```

## Using Fine-Tuned Models

### Automatic Usage

Once trained, the system automatically uses the fine-tuned model:

```python
# In retriever or chat endpoint
from llmservice.query_optimizer import optimize_user_query

# User's original query
user_query = "what are transformers?"

# Optimized query (uses fine-tuned model if available)
optimized = optimize_user_query(user_query)
# Output: "Explain the transformer architecture in neural networks,
#          including self-attention mechanisms and their applications in NLP"
```

### Manual Loading

```python
from llmservice.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer()
optimizer.load_model("./models/lora_query_optimizer")

# Optimize single query
optimized = optimizer.optimize_query("how does attention work?")

# Batch optimization
queries = ["what is LoRA?", "explain fine-tuning"]
optimized_batch = optimizer.batch_optimize(queries)
```

## Training Data Format

The system generates training data in this format:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a query optimization assistant..."
      },
      {
        "role": "user",
        "content": "Optimize this query: what are transformers?"
      },
      {
        "role": "assistant",
        "content": "Explain the transformer neural network architecture, focusing on self-attention mechanisms, encoder-decoder structure, and applications in natural language processing"
      }
    ],
    "metadata": {
      "context": "...",
      "chunk_hash": "abc123",
      "generated_at": "2026-02-25T10:30:00"
    }
  }
]
```

## Hardware Requirements

### Minimum (Lightweight Config)

- **GPU**: 8GB VRAM (RTX 3060, RTX 4060)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space
- **Training Time**: 1-2 hours for 500 examples

### Recommended (Performance Config)

- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space
- **Training Time**: 30-45 minutes for 500 examples

### Cloud Alternatives

- **Google Colab**: Free T4 GPU (15GB VRAM)
- **Lambda Labs**: Starting at $0.50/hr
- **AWS SageMaker**: P3 instances
- **Hugging Face Spaces**: A10G GPU

## Monitoring Training

### TensorBoard

```bash
# View training metrics in real-time
tensorboard --logdir ./logs/lora_training
```

### Training Logs

Logs are saved to: `./logs/lora_training/training_YYYYMMDD_HHMMSS.log`

### Key Metrics

- **Training Loss**: Should decrease over time (< 1.0 is good)
- **Evaluation Loss**: Should decrease without increasing (no overfitting)
- **Learning Rate**: Follows cosine schedule
- **GPU Memory**: Monitor to prevent OOM errors

## Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size
LORA_BATCH_SIZE = 2

# Reduce sequence length
LORA_MAX_SEQ_LENGTH = 256

# Reduce LoRA rank
LORA_RANK = 8

# Enable gradient checkpointing (already on by default)
config.gradient_checkpointing = True
```

### Slow Training

```python
# Increase batch size (if memory allows)
LORA_BATCH_SIZE = 8

# Reduce gradient accumulation
config.gradient_accumulation_steps = 2

# Use bf16 (Ampere GPUs and newer)
config.bf16 = True
config.fp16 = False
```

### Poor Results

1. **More Training Data**: Generate more examples (1000+ recommended)
2. **More Epochs**: Increase from 3 to 5-10
3. **Higher LoRA Rank**: Increase from 16 to 32 or 64
4. **Learning Rate**: Try 1e-4 or 3e-4
5. **Better Base Model**: Try Llama-2-13B instead of 7B

## Model Management

### List Trained Models

```bash
ls -lh ./models/lora_query_optimizer/
```

### Load Specific Checkpoint

```python
optimizer.load_model("./models/lora_query_optimizer/checkpoint-500")
```

### Export for Deployment

```python
# Merge LoRA adapters into base model (for faster inference)
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(base_model, "./models/lora_query_optimizer")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./models/merged_optimizer")
```

## Best Practices

1. **Start Small**: Begin with lightweight config and 200-300 examples
2. **Monitor Validation**: Watch for overfitting (eval loss increases)
3. **Iterate**: Train → Evaluate → Adjust → Retrain
4. **Domain-Specific**: More domain documents = better optimization
5. **Regular Updates**: Retrain as you add new document types
6. **Version Control**: Keep track of which model version works best

## API Integration

### REST API Endpoint (Future Enhancement)

```python
# Add to main.py
@app.post("/api/optimize-query")
async def optimize_query_endpoint(query: str):
    from llmservice.query_optimizer import optimize_user_query
    optimized = optimize_user_query(query)
    return {"original": query, "optimized": optimized}
```

## Advanced Topics

### Custom Target Modules

```python
config.lora_target_modules = [
    "q_proj",  # Query projection
    "v_proj",  # Value projection
    "k_proj",  # Key projection
    "o_proj",  # Output projection
]
```

### Multi-GPU Training

```python
# Use FSDP or DeepSpeed for multiple GPUs
config.fsdp = "full_shard auto_wrap"
config.fsdp_transformer_layer_cls_to_wrap = "MistralDecoderLayer"
```

### Custom Data Augmentation

```python
from training.data_processor import TrainingDataProcessor

processor = TrainingDataProcessor(
    system_prompt="Custom system prompt for your domain..."
)

dataset = processor.create_dataset(
    "training_data.json",
    train_test_split=0.15,
    shuffle=True
)
```

## Support and Resources

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See main README.md
- **Examples**: Check `backend/Scripts/training/` for notebooks
- **Community**: Join our Discord/Slack for help

## License

Same as parent project (see main LICENSE file).
