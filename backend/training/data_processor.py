"""
Training Data Processor
Handles conversion of generated training data to HuggingFace datasets format.
"""

import json
import os
from typing import List, Dict, Optional, Union
from datasets import Dataset, DatasetDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TrainingDataProcessor:
    """
    Processes training data for LoRA fine-tuning.
    Converts JSON training examples into HuggingFace datasets.
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the data processor
        
        Args:
            system_prompt: System prompt to prepend to all training examples
        """
        self.system_prompt = system_prompt or (
            "You are a query optimization assistant. Transform user queries into more effective, "
            "precise queries for information retrieval in a RAG system. Make queries more specific, "
            "add relevant context, and structure them for better semantic search."
        )
    
    def load_training_data(self, file_path: Union[str, Path]) -> List[Dict]:
        """
        Load training data from JSON file
        
        Args:
            file_path: Path to JSON training data file
        
        Returns:
            List of training examples
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples from {file_path}")
        return data
    
    def format_example_for_training(self, example: Dict) -> Dict:
        """
        Format a single example for instruction fine-tuning
        
        Args:
            example: Training example with 'messages' or individual fields
        
        Returns:
            Formatted example with 'text' field for training
        """
        # Handle different input formats
        if 'messages' in example:
            messages = example['messages']
            # Extract user query and assistant response
            user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
            assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
        elif 'unoptimized_query' in example and 'optimized_query' in example:
            user_msg = f"Optimize this query: {example['unoptimized_query']}"
            assistant_msg = example['optimized_query']
        else:
            raise ValueError("Example must contain either 'messages' or 'unoptimized_query'/'optimized_query'")
        
        # Format for instruction tuning (using Alpaca-style format)
        formatted_text = self._format_instruction_text(user_msg, assistant_msg)
        
        return {
            'text': formatted_text,
            'input': user_msg,
            'output': assistant_msg,
        }
    
    def _format_instruction_text(self, instruction: str, response: str) -> str:
        """
        Format instruction and response into training text
        Uses Mistral/Llama chat template format
        
        Args:
            instruction: User instruction/query
            response: Model response
        
        Returns:
            Formatted text string
        """
        # Mistral Instruct format
        formatted = f"""<s>[INST] {self.system_prompt}

{instruction} [/INST] {response}</s>"""
        return formatted
    
    def create_dataset(
        self, 
        data: Union[List[Dict], str, Path],
        train_test_split: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> DatasetDict:
        """
        Create HuggingFace dataset from training data
        
        Args:
            data: Training data (list of dicts or path to JSON file)
            train_test_split: Fraction of data to use for testing
            shuffle: Whether to shuffle data before splitting
            seed: Random seed for shuffling
        
        Returns:
            DatasetDict with train and test splits
        """
        # Load data if path is provided
        if isinstance(data, (str, Path)):
            data = self.load_training_data(data)
        
        # Format all examples
        formatted_examples = []
        for example in data:
            try:
                formatted = self.format_example_for_training(example)
                formatted_examples.append(formatted)
            except Exception as e:
                logger.warning(f"Skipping example due to formatting error: {e}")
                continue
        
        logger.info(f"Successfully formatted {len(formatted_examples)} examples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(seed=seed)
        
        # Split into train and test
        if train_test_split > 0:
            split_dataset = dataset.train_test_split(test_size=train_test_split, seed=seed)
            dataset_dict = DatasetDict({
                'train': split_dataset['train'],
                'test': split_dataset['test']
            })
        else:
            dataset_dict = DatasetDict({
                'train': dataset
            })
        
        logger.info(f"Created dataset with {len(dataset_dict['train'])} training examples")
        if 'test' in dataset_dict:
            logger.info(f"and {len(dataset_dict['test'])} test examples")
        
        return dataset_dict
    
    def prepare_for_sft(
        self,
        dataset: Dataset,
        tokenizer,
        max_length: int = 512
    ) -> Dataset:
        """
        Prepare dataset for Supervised Fine-Tuning (SFT)
        
        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        
        Returns:
            Tokenized dataset ready for training
        """
        def tokenize_function(examples):
            """Tokenize the text field"""
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                padding='max_length',
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        return tokenized_dataset
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, any]:
        """
        Validate dataset and return statistics
        
        Args:
            dataset: Dataset to validate
        
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'num_examples': len(dataset),
            'columns': dataset.column_names,
            'features': dataset.features,
        }
        
        # Check for required fields
        required_fields = ['text']
        missing_fields = [f for f in required_fields if f not in dataset.column_names]
        
        if missing_fields:
            stats['valid'] = False
            stats['error'] = f"Missing required fields: {missing_fields}"
        else:
            stats['valid'] = True
        
        # Sample statistics
        if len(dataset) > 0:
            first_example = dataset[0]
            stats['first_example_length'] = len(first_example.get('text', ''))
            
            # Average text length
            text_lengths = [len(ex['text']) for ex in dataset.select(range(min(100, len(dataset))))]
            stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
            stats['max_text_length'] = max(text_lengths)
            stats['min_text_length'] = min(text_lengths)
        
        return stats
    
    def save_dataset(self, dataset: DatasetDict, output_dir: Union[str, Path]):
        """
        Save processed dataset to disk
        
        Args:
            dataset: DatasetDict to save
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset.save_to_disk(str(output_dir))
        logger.info(f"Saved dataset to {output_dir}")
    
    def load_dataset(self, dataset_dir: Union[str, Path]) -> DatasetDict:
        """
        Load processed dataset from disk
        
        Args:
            dataset_dir: Directory containing saved dataset
        
        Returns:
            Loaded DatasetDict
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        from datasets import load_from_disk
        dataset = load_from_disk(str(dataset_dir))
        logger.info(f"Loaded dataset from {dataset_dir}")
        return dataset


def convert_training_json_to_dataset(
    json_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    train_test_split: float = 0.1
) -> DatasetDict:
    """
    Convenience function to convert training JSON to HuggingFace dataset
    
    Args:
        json_file: Path to training JSON file
        output_dir: Optional directory to save processed dataset
        train_test_split: Fraction for test split
    
    Returns:
        Processed DatasetDict
    """
    processor = TrainingDataProcessor()
    dataset = processor.create_dataset(json_file, train_test_split=train_test_split)
    
    if output_dir:
        processor.save_dataset(dataset, output_dir)
    
    return dataset
