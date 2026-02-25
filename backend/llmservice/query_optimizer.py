"""
Query Optimizer Service
Uses fine-tuned LoRA model to optimize user queries for better retrieval.
"""

import torch
from typing import Optional
from pathlib import Path
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config.settings import (
    USE_FINETUNED_OPTIMIZER,
    FINETUNED_OPTIMIZER_PATH,
    LORA_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    Query optimizer using fine-tuned LoRA model
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        
        # Auto-load if fine-tuned model is available
        if USE_FINETUNED_OPTIMIZER and FINETUNED_OPTIMIZER_PATH:
            try:
                self.load_model(FINETUNED_OPTIMIZER_PATH)
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model: {e}")
    
    def load_model(self, adapter_path: str):
        """
        Load fine-tuned LoRA model
        
        Args:
            adapter_path: Path to LoRA adapters
        """
        adapter_path = Path(adapter_path)
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        logger.info(f"Loading query optimizer from: {adapter_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            LORA_MODEL_NAME,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self.model.eval()
        
        self.loaded = True
        logger.info("Query optimizer loaded successfully")
    
    def optimize_query(self, query: str, max_length: int = 256) -> str:
        """
        Optimize a user query for better retrieval
        
        Args:
            query: Original user query
            max_length: Maximum length for generated query
        
        Returns:
            Optimized query string
        """
        if not self.loaded:
            logger.warning("Model not loaded, returning original query")
            return query
        
        # Format input
        prompt = self._format_prompt(query)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract optimized query from response
        optimized_query = self._extract_optimized_query(generated_text, prompt)
        
        logger.info(f"Original query: {query}")
        logger.info(f"Optimized query: {optimized_query}")
        
        return optimized_query
    
    def _format_prompt(self, query: str) -> str:
        """Format query for the model"""
        system_prompt = (
            "You are a query optimization assistant. Transform user queries into more effective, "
            "precise queries for information retrieval in a RAG system."
        )
        
        prompt = f"""<s>[INST] {system_prompt}

Optimize this query: {query} [/INST] """
        
        return prompt
    
    def _extract_optimized_query(self, generated_text: str, original_prompt: str) -> str:
        """Extract the optimized query from generated text"""
        # Remove the prompt from the generated text
        if original_prompt in generated_text:
            response = generated_text.replace(original_prompt, "").strip()
        else:
            response = generated_text.strip()
        
        # Remove special tokens
        response = response.replace("</s>", "").replace("<s>", "").strip()
        
        # If response is empty or too similar to input, return as is
        if not response or len(response) < 3:
            return generated_text.split("[/INST]")[-1].strip() if "[/INST]" in generated_text else generated_text
        
        return response
    
    def is_available(self) -> bool:
        """Check if optimizer is loaded and available"""
        return self.loaded
    
    def batch_optimize(self, queries: list[str], max_length: int = 256) -> list[str]:
        """
        Optimize multiple queries in batch
        
        Args:
            queries: List of queries to optimize
            max_length: Maximum length for generated queries
        
        Returns:
            List of optimized queries
        """
        if not self.loaded:
            logger.warning("Model not loaded, returning original queries")
            return queries
        
        optimized = []
        for query in queries:
            try:
                optimized_query = self.optimize_query(query, max_length)
                optimized.append(optimized_query)
            except Exception as e:
                logger.error(f"Error optimizing query '{query}': {e}")
                optimized.append(query)  # Fallback to original
        
        return optimized


# Global singleton instance
_query_optimizer_instance: Optional[QueryOptimizer] = None


def get_query_optimizer() -> QueryOptimizer:
    """
    Get or create global query optimizer instance
    
    Returns:
        QueryOptimizer instance
    """
    global _query_optimizer_instance
    
    if _query_optimizer_instance is None:
        _query_optimizer_instance = QueryOptimizer()
    
    return _query_optimizer_instance


def optimize_user_query(query: str) -> str:
    """
    Convenience function to optimize a single query
    
    Args:
        query: User query to optimize
    
    Returns:
        Optimized query (or original if optimizer not available)
    """
    optimizer = get_query_optimizer()
    
    if optimizer.is_available():
        return optimizer.optimize_query(query)
    else:
        return query
