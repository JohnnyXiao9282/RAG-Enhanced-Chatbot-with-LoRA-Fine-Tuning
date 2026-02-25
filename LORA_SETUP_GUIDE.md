# LoRA Fine-Tuning Implementation - Getting Started Guide

## 🎉 Implementation Complete!

Your RAG system now has full LoRA fine-tuning capabilities for query optimization! This guide will help you get started.

## 📋 What Was Implemented

### Core Components
✅ **Training Infrastructure** (`backend/training/`)
- `lora_trainer.py` - Main training logic with 4-bit quantization support
- `model_config.py` - Flexible configuration system (lightweight/performance)
- `data_processor.py` - Automatic dataset formatting and processing
- `query_optimizer.py` - Inference service for using fine-tuned models

✅ **Integration**
- Auto-training during document upload (when enabled)
- Seamless integration with existing training data generation
- Automatic model loading for query optimization
- Settings management through `config/settings.py`

✅ **Dependencies**
- All required packages added to `requirements.txt`
- Support for PyTorch, Transformers, PEFT, TRL, and more

## 🚀 Quick Start (5 Steps)

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note**: This will install ~5GB of packages. Ensure you have:
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 20GB free disk space

### Step 2: Configure Settings

Edit `backend/config/settings.py`:

```python
# Enable LoRA training
LORA_ENABLED = True

# Use lightweight config for 8GB VRAM (recommended for most users)
LORA_USE_LIGHTWEIGHT_CONFIG = True

# Choose base model (default is Mistral-7B)
LORA_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Training parameters
LORA_EPOCHS = 3                    # Number of training epochs
LORA_MIN_TRAINING_EXAMPLES = 100   # Minimum examples needed
```

### Step 3: Upload Documents with Training Enabled

**Via UI:**
1. Go to Upload panel in the web interface
2. Select your PDF documents
3. ✅ **Check "Enable model training with uploaded documents"**
4. Click Upload

**Programmatically:**
```python
from ingestion.ingestion import Ingestaion

ingestion = Ingestaion(
    parser="MistralOCR",
    chunker="groupEmbeddingAlgo",
    encoder=encoder
)

ingestion.ingest_files(
    file_path_lists=["path/to/document.pdf"],
    save_json=True,
    train_query_opt=True  # Enable LoRA training
)
```

### Step 4: Wait for Training

Training happens automatically in the background:
1. **Data Generation** (~5-10 min): Creates query optimization pairs
2. **Model Download** (~5-10 min): Downloads base model (first time only)
3. **LoRA Training** (~30-60 min): Fine-tunes the model

Monitor progress in the logs: `logs/lora_training/training_YYYYMMDD_HHMMSS.log`

### Step 5: Use Fine-Tuned Model

Once training completes, the system automatically uses the fine-tuned model:

```python
from llmservice.query_optimizer import optimize_user_query

# User's query
query = "what are transformers?"

# Automatically optimized using fine-tuned model
optimized = optimize_user_query(query)
# Output: "Explain the transformer architecture in neural networks, 
#          including self-attention mechanisms and their applications in NLP"
```

## 💻 Hardware Requirements

### Minimum Setup (Lightweight Config)
- **GPU**: 8GB VRAM (RTX 3060, RTX 4060, or better)
- **RAM**: 16GB system memory
- **Storage**: 20GB free space
- **Training Time**: 1-2 hours for 500 examples

### Recommended Setup
- **GPU**: 16GB+ VRAM (RTX 3080, RTX 4080, A4000)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space
- **Training Time**: 30-45 minutes for 500 examples

### No GPU?
Use cloud options:
- **Google Colab**: Free T4 GPU (15GB VRAM) - [colab.research.google.com](https://colab.research.google.com)
- **Lambda Labs**: Starting at $0.50/hr - [lambdalabs.com](https://lambdalabs.com)
- **Paperspace**: Gradient notebooks - [paperspace.com](https://paperspace.com)

## 🎯 Usage Examples

### Example 1: Basic Training

```python
from training.lora_trainer import quick_train

# Train with default settings
metrics = quick_train(
    training_data_path="./data/training_data/training_data_20260225.json",
    output_dir="./models/my_optimizer",
    epochs=3,
    use_lightweight=True
)

print(f"Training complete! Loss: {metrics['train_loss']}")
```

### Example 2: Custom Configuration

```python
from training.lora_trainer import LoRATrainer
from training.model_config import LoRAConfig

# Create custom config
config = LoRAConfig(
    base_model_name="mistralai/Mistral-7B-Instruct-v0.2",
    lora_r=32,              # Higher rank = more capacity
    lora_alpha=64,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    output_dir="./models/custom_optimizer"
)

# Train
trainer = LoRATrainer(config)
metrics = trainer.train_from_json("./data/training_data/my_data.json")
```

### Example 3: Batch Query Optimization

```python
from llmservice.query_optimizer import get_query_optimizer

optimizer = get_query_optimizer()

# Optimize multiple queries
queries = [
    "what is LoRA?",
    "how does attention work?",
    "explain transformer architecture"
]

optimized = optimizer.batch_optimize(queries)

for orig, opt in zip(queries, optimized):
    print(f"Original:  {orig}")
    print(f"Optimized: {opt}\n")
```

## 📊 Monitoring Training

### View TensorBoard Metrics

```bash
# Start TensorBoard
tensorboard --logdir ./logs/lora_training

# Open in browser: http://localhost:6006
```

### Check Training Logs

```bash
# View latest log
tail -f ./logs/lora_training/training_*.log

# View all logs
ls -lh ./logs/lora_training/
```

### Key Metrics to Watch
- **Training Loss**: Should decrease to < 1.0
- **Eval Loss**: Should decrease without increasing (no overfitting)
- **GPU Memory**: Monitor to avoid OOM errors

## 🔧 Troubleshooting

### Problem: Out of Memory (OOM)

**Solution 1**: Reduce batch size in `settings.py`:
```python
LORA_BATCH_SIZE = 2
```

**Solution 2**: Reduce sequence length:
```python
LORA_MAX_SEQ_LENGTH = 256
```

**Solution 3**: Reduce LoRA rank:
```python
LORA_RANK = 8
```

### Problem: Training Too Slow

**Solution 1**: Increase batch size (if memory allows):
```python
LORA_BATCH_SIZE = 8
```

**Solution 2**: Use performance config:
```python
LORA_USE_LIGHTWEIGHT_CONFIG = False
```

### Problem: Poor Optimization Results

**Solutions**:
1. **More Data**: Generate 500-1000+ training examples
2. **More Epochs**: Increase to 5-10 epochs
3. **Higher Rank**: Set `LORA_RANK = 32` or `64`
4. **Better Base Model**: Try Llama-2 or larger models

### Problem: Model Not Being Used

**Check**:
```python
from llmservice.query_optimizer import get_query_optimizer

optimizer = get_query_optimizer()
print(f"Model loaded: {optimizer.is_available()}")
```

**Enable manually in `settings.py`**:
```python
USE_FINETUNED_OPTIMIZER = True
FINETUNED_OPTIMIZER_PATH = "./models/lora_query_optimizer"
```

## 📁 File Structure

```
backend/
├── training/                    # LoRA training module
│   ├── __init__.py             # Package initialization
│   ├── lora_trainer.py         # Main training logic (380 lines)
│   ├── model_config.py         # Configuration system (200 lines)
│   ├── data_processor.py       # Dataset processing (260 lines)
│   └── README.md               # Detailed documentation
│
├── llmservice/
│   ├── query_optimizer.py      # Inference service (200 lines)
│   └── llmhelper.py            # Updated with LoRA integration
│
├── config/
│   └── settings.py             # Added LoRA settings
│
├── Scripts/training/
│   └── lora_quickstart.ipynb   # Interactive tutorial
│
├── models/                      # Trained models saved here
│   └── lora_query_optimizer/   # Default output directory
│
└── logs/                        # Training logs
    └── lora_training/           # TensorBoard logs
```

## 🎓 Learning Resources

### Documentation
- **Training README**: `backend/training/README.md` - Comprehensive guide
- **Quick Start Notebook**: `backend/Scripts/training/lora_quickstart.ipynb`
- **Code Comments**: All modules have detailed docstrings

### External Resources
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research
- [PEFT Documentation](https://huggingface.co/docs/peft) - HuggingFace PEFT library
- [TRL Documentation](https://huggingface.co/docs/trl) - Transformer Reinforcement Learning

## 🔄 Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Upload Documents                       │
│              (with "Train Model" enabled)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           2. Automatic Data Generation (GPT-4)              │
│    Creates query optimization pairs from documents          │
│              (unoptimized → optimized)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              3. LoRA Fine-Tuning (Automatic)                │
│    - Downloads base model (Mistral-7B)                      │
│    - Applies LoRA adapters (trains ~1% of parameters)       │
│    - Saves fine-tuned model                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         4. Query Optimization (Automatic in Chat)           │
│    User query → Fine-tuned model → Optimized query          │
│    → Better retrieval → Better answers                       │
└─────────────────────────────────────────────────────────────┘
```

## 🚦 Next Steps

1. **Test the System**
   ```bash
   # Run the quick start notebook
   jupyter notebook backend/Scripts/training/lora_quickstart.ipynb
   ```

2. **Upload Real Documents**
   - Start with 5-10 documents from your domain
   - Enable training in the UI
   - Wait for training to complete (~1-2 hours)

3. **Evaluate Performance**
   - Test queries before and after training
   - Compare retrieval accuracy
   - Monitor user satisfaction

4. **Iterate and Improve**
   - Add more documents over time
   - Retrain periodically
   - Adjust hyperparameters based on results

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in `./logs/lora_training/`
3. Consult `backend/training/README.md` for detailed docs
4. Open a GitHub issue with error details

## 🎊 Congratulations!

You now have a complete LoRA fine-tuning system integrated into your RAG chatbot. The system will:
- ✅ Automatically generate training data from your documents
- ✅ Fine-tune models with minimal GPU memory
- ✅ Optimize queries for better retrieval
- ✅ Continuously improve as you add more documents

Happy fine-tuning! 🚀
