from config.settings import *
from utils.logger import Logger
import tiktoken
import openai
from llmservice.adaptiveJsonExtractor import AdaptiveJsonExtractor
from llmservice.prompts import PROMPTS
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from openai import AsyncOpenAI
import asyncio
import json
from datetime import datetime
import aiofiles

load_dotenv(dotenv_path="./.env")

class LLmHelper:
    
    def __init__(self,  generation_model):
        self.generation_model = generation_model
        self.logger = Logger(name="RAGLogger").get_logger()
        self.max_tokens = MAX_ALLOWED_TOKENS.get(self.generation_model, None)
        self.token_encoding_obj = tiktoken.encoding_for_model(self.generation_model)
        self.adaptive_json_extractor = AdaptiveJsonExtractor()
        self.openai_api_key = os.getenv('OPEN_AI_API_KEY')
        os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.openai_client = openai.OpenAI()
    
    def idealChunkTokens(self, top_k, chunks_coverage):
        """Calculate ideal chunk token size based on model and configuration."""
        try:
            generation_model = self.generation_model
            self.logger.info("Calculating ideal chunk tokens for model: %s", self.generation_model)
            
            total_chunks = top_k
            max_tokens = self.max_tokens

            if max_tokens is None:
                self.logger.error("Model %s not defined in MAX_ALLOWED_TOKENS configuration", generation_model)
                raise ValueError(f"Model {generation_model} not defined in the configurations")
            
            if total_chunks == 0:
                self.logger.error("TOTAL_CHUNKS_CONSIDERED is 0, would cause division by zero")
                raise ValueError("TOTAL_CHUNKS_CONSIDERED cannot be 0")
            
            ideal_tokens = (max_tokens * (chunks_coverage / 100)) / total_chunks
            self.logger.info("Calculated ideal chunk tokens: %d", int(ideal_tokens))
            return int(ideal_tokens)
            
        except Exception as e:
            self.logger.error("Error calculating ideal chunk tokens: %s", str(e))
            raise
    
    def getTokens(self, sent):
        """Get token count for a sentence."""
        try:
            if not sent:
                return 0
            
            tokens = len(self.token_encoding_obj.encode(sent))
            self.logger.debug("Sentence has %d tokens", tokens)
            return tokens
            
        except Exception as e:
            self.logger.error("Error counting tokens: %s", str(e))
            raise

    async def generate_training_examples(self, chunks, openAi_client, max_examples_per_chunks = 4):
        training_data = []
        for i, chunk in enumerate(chunks):
            chunk_content = chunk['content']
            chunk_id = chunk['chunk_id']
            examples = await self._generate_examples_for_chunk(chunk_content, chunk_id, max_examples_per_chunks)
            training_data.extend(examples)
        
        return training_data

    async def _generate_examples_for_chunk(self, chunk_content, chunk_id, openAi_client, num_examples):
        prompt = PROMPTS['Training_Data_Generator']
        prompt = prompt.replace("{num_examples}", num_examples).replace("{chunk}", '\n'+chunk_content)
        
        try:
            response = await openAi_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating training data for query optimization models. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            examples = json.loads(content)
            
            for example in examples:
                example.update({
                    "chunk_hash": chunk_id,
                    "generated_at": datetime.now().isoformat()
                })
            return examples
        except Exception as e:
            pass
    
    async def create_training_dataset(self, chunks, openAi_client, output_file):
        training_examples = generate_training_examples(self, chunks, openAi_client)
        formatted_data = []
        for example in training_examples:
            formatted_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a query optimization assistant. Transform user queries into more effective, precise queries for information retrieval."
                    },
                    {
                        "role": "user", 
                        "content": f"Optimize this query: {example['unoptimized_query']}"
                    },
                    {
                        "role": "assistant",
                        "content": example['optimized_query']
                    }
                ],
                "metadata": {
                    "context": example.get('context', ''),
                    "chunk_hash": example.get('chunk_hash', ''),
                    "generated_at": example.get('generated_at', '')
                }
            }
            formatted_data.append(formatted_example)
        async with aiofiles.open(output_file, 'w') as f:
            await f.write(json.dumps(formatted_data, indent = 2))
        
        return output_file
    
    async def trainOptimizer(self, chunks, output_file):
        """
        Train query optimizer with LoRA fine-tuning
        1. Generate training data using GPT-4
        2. Optionally fine-tune model with LoRA if enabled
        """
        openAi_client = AsyncOpenAI(api_key = self.openai_api_key)
        
        # Generate training data
        self.logger.info("Generating training data...")
        output_file = await self.create_training_dataset(chunks, openAi_client, output_file)
        self.logger.info(f"Training data saved to: {output_file}")
        
        # Check if we have enough examples for LoRA training
        if not LORA_ENABLED:
            self.logger.info("LoRA training is disabled in settings")
            return output_file
        
        try:
            # Count examples in generated file
            with open(output_file, 'r') as f:
                training_data = json.load(f)
                num_examples = len(training_data)
            
            self.logger.info(f"Generated {num_examples} training examples")
            
            if num_examples < LORA_MIN_TRAINING_EXAMPLES:
                self.logger.warning(
                    f"Not enough training examples ({num_examples}/{LORA_MIN_TRAINING_EXAMPLES}). "
                    "Skipping LoRA fine-tuning."
                )
                return output_file
            
            # Import LoRA trainer
            from training.lora_trainer import quick_train
            from training.model_config import get_lightweight_config, get_default_config
            
            self.logger.info("Starting LoRA fine-tuning...")
            
            # Setup config
            if LORA_USE_LIGHTWEIGHT_CONFIG:
                config = get_lightweight_config()
            else:
                config = get_default_config()
            
            # Update config from settings
            config.base_model_name = LORA_MODEL_NAME
            config.output_dir = LORA_OUTPUT_DIR
            config.num_train_epochs = LORA_EPOCHS
            config.per_device_train_batch_size = LORA_BATCH_SIZE
            config.lora_r = LORA_RANK
            config.lora_alpha = LORA_ALPHA
            config.max_seq_length = LORA_MAX_SEQ_LENGTH
            
            # Train model
            metrics = quick_train(
                training_data_path=output_file,
                output_dir=config.output_dir,
                model_name=config.base_model_name,
                epochs=config.num_train_epochs,
                use_lightweight=LORA_USE_LIGHTWEIGHT_CONFIG
            )
            
            self.logger.info("LoRA fine-tuning completed!")
            self.logger.info(f"Model saved to: {config.output_dir}")
            self.logger.info(f"Training metrics: {metrics}")
            
            # Update settings to use the fine-tuned model
            global FINETUNED_OPTIMIZER_PATH, USE_FINETUNED_OPTIMIZER
            FINETUNED_OPTIMIZER_PATH = config.output_dir
            USE_FINETUNED_OPTIMIZER = True
            
            self.logger.info("Fine-tuned model is now available for query optimization")
            
        except Exception as e:
            self.logger.error(f"Error during LoRA training: {str(e)}")
            self.logger.warning("Continuing without fine-tuned model")
        
        return output_file

    