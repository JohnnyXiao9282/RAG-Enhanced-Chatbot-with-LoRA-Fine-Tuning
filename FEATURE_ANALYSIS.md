# RAG-Enhanced Chatbot: Comprehensive Feature Analysis

## Table of Contents
1. [Multi-LLM Jury System](#1-multi-llm-jury-system)
2. [Advanced Chunking Algorithms](#2-advanced-chunking-algorithms)
3. [Intelligent Re-ranking System](#3-intelligent-re-ranking-system)
4. [Chain of Thoughts Processing](#4-chain-of-thoughts-processing)
5. [Adaptive JSON Extraction](#5-adaptive-json-extraction)
6. [Image Integration System](#6-image-integration-system)
7. [Document Processing Pipeline](#7-document-processing-pipeline)
8. [Vector Database Integration](#8-vector-database-integration)
9. [Response Generation Modes](#9-response-generation-modes)
10. [Advanced Configuration Options](#10-advanced-configuration-options)

---

## 1. Multi-LLM Jury System

### Overview
The Multi-LLM Jury System is a sophisticated ensemble approach that simulates a jury deliberation process using multiple AI models to improve response quality and accuracy.

### Architecture Components

#### 1.1 Processing Modes
```python
class ProcessingMode(Enum):
    SINGLE_LLM = "single"           # Single model processing
    MULTI_LLM_JURY = "jury"         # Jury-based ensemble
    CHAIN_OF_THOUGHTS = "cot"       # Iterative reasoning
```

#### 1.2 Jury Workflow

**Phase 1: Master Initial Response**
- **Purpose**: Generate initial comprehensive answer
- **Model**: Primary LLM (configurable)
- **Output**: Structured response with confidence scoring
- **Key Features**:
  - Context-aware answer generation
  - Confidence score assessment (1-10 scale)
  - Review necessity evaluation
  - Image tag integration

**Phase 2: Jury Deliberation**
- **Iterations**: Configurable (default: 3 iterations)
- **Models**: Alternating between different providers
  - Provider rotation: `["gemini", "openai"]`
  - Model rotation: `["gemini-pro", "gpt-4o"]`
- **Process**: Each jury member reviews and improves previous response

**Phase 3: Master Evaluation**
- **Purpose**: Synthesize all jury inputs into final answer
- **Process**: Combine insights while maintaining accuracy
- **Output**: Optimized final response

### 1.3 Detailed Jury Prompts

#### Master Initial Response Prompt
```json
{
  "answer": "Comprehensive answer with image tags",
  "reasoning": "Step-by-step derivation process",
  "confidence_score": 8,
  "needs_review": true,
  "review_reason": "Complexity requires multiple perspectives",
  "relevant_image_tags": ["hash_1", "hash_2"],
  "context_coverage": "complete/partial/limited"
}
```

#### Jury Member Prompt (SlaveRecursivePrompt)
**Actions Available**:
- **ENHANCE**: Add valuable information or alternative perspectives
- **CORRECT**: Fix factual errors or misinterpretations
- **RESTRUCTURE**: Improve organization or clarity
- **VALIDATE**: Confirm answer is comprehensive and accurate

**Evaluation Criteria**:
- Factual accuracy against context
- Completeness of answer
- Clarity and organization
- Proper image reference integration
- Alternative perspectives or insights

#### Master Opinion Check Prompt
**Decision Criteria**:
- Did jury add valuable new insights?
- Are there remaining gaps or uncertainties?
- Would another perspective help?
- Is the answer now comprehensive enough?

**Output Format**:
```json
{
  "continue_pipeline": true,
  "reasoning": "Why continue/stop the pipeline",
  "current_quality_score": 8,
  "improvement_needed": "What aspects need more work",
  "estimated_remaining_value": "high/medium/low"
}
```

#### Master Final Evaluation Prompt
**Synthesis Guidelines**:
- Preserve factual accuracy from context
- Include best insights from each LLM
- Resolve contradictions logically
- Maintain image tag integration
- Ensure coherent, comprehensive response

### 1.4 Jury System Benefits

**Quality Improvements**:
- **Accuracy**: Multiple perspectives reduce bias
- **Completeness**: Different models catch different aspects
- **Robustness**: Error correction through consensus
- **Depth**: Enhanced reasoning through iteration

**Adaptive Behavior**:
- **Early Termination**: Stop when quality threshold met
- **Dynamic Iteration**: Continue until optimal result
- **Quality Assessment**: Real-time evaluation of improvements

---

## 2. Advanced Chunking Algorithms

### 2.1 Group Embedding Algorithm (GroupEAlgo)

#### Core Concept
The GroupEAlgo uses semantic similarity to determine optimal chunk boundaries, ensuring related content stays together while maintaining manageable chunk sizes.

#### Algorithm Parameters
```python
# Configuration Settings
DEFAULT_CHUNK_LENGTH = 1000    # Maximum tokens per chunk
DEFAULT_OVER_LAP = 50          # Token overlap between chunks
PDF_SPLIT_SIZE = 5             # Pages per PDF split
PAGE_COMBO = 2                 # Pages combined for analysis
SENTENCE_COMBO = 4             # Sentences combined for embedding
```

#### Chunking Process

**Step 1: Page-Level Processing**
- Combine multiple pages (PAGE_COMBO = 2)
- Generate page-level embeddings
- Identify semantic boundaries

**Step 2: Sentence-Level Analysis**
- Group sentences (SENTENCE_COMBO = 4)
- Calculate sentence embeddings
- Determine semantic cohesion

**Step 3: Chunk Formation**
- **Cohesion Test**: `stayTogether(e1, e2, e3, E)`
  - e1: Current chunk embedding
  - e2: Next sentence group embedding
  - e3: Combined embedding
  - E: Page-level embedding
- **Size Constraint**: `getTokens(combo_chunk) <= max_chunk_length`

#### Semantic Cohesion Algorithm
```python
def stayTogether(self, e1, e2, e3, E):
    # Calculate cosine similarities
    sim_1_2 = cosine_similarity([e1], [e2])[0][0]
    sim_1_3 = cosine_similarity([e1], [e3])[0][0]
    sim_2_3 = cosine_similarity([e2], [e3])[0][0]
    
    # Determine if chunks should stay together
    return (sim_1_3 > sim_1_2) and (sim_2_3 > threshold)
```

### 2.2 Chunk Metadata Structure
```json
{
  "chunk_id": "uuid",
  "file_info": {...},
  "chunk_info": {
    "chunk_index": 0,
    "encoder": "sentence_encoder",
    "language": "en",
    "chunk_size": 850,
    "chunk_type": "sentence_chunk",
    "prev_chunk_id": "uuid",
    "next_chunk_id": "uuid"
  },
  "content": "Chunk text content",
  "embedding": "vector_embedding",
  "semantic_info": {
    "keywords": ["entity1", "entity2"]
  },
  "media_ref": {
    "images": [...]
  }
}
```

---

## 3. Intelligent Re-ranking System

### 3.1 Multi-Factor Scoring Algorithm

The re-ranking system combines three different scoring methods with configurable weights:

#### Scoring Components
```python
RERANKING_PARAMETERS_PERCENT = {
    'tfidf': 20,      # Term frequency-inverse document frequency
    'kw': 30,         # Keyword overlap scoring
    'vec_sim': 50     # Vector similarity scoring
}
```

### 3.2 TF-IDF Scoring
**Purpose**: Term-based relevance scoring
**Process**:
1. Create TF-IDF matrix from query + chunks
2. Calculate cosine similarity between query and each chunk
3. Normalize scores using MinMaxScaler
4. Apply weight percentage to final score

```python
def getTfIDFScore(self, query, chunks, percentage_val):
    corpus = [query] + [chunk['content'] for chunk in chunks]
    tfidf_mat = self.vectorizer.fit_transform(corpus)
    tfidf_sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
    # Normalize and apply weight
```

### 3.3 Keyword Overlap Scoring
**Purpose**: Entity-based relevance matching
**Process**:
1. Extract named entities from query using spaCy
2. Compare with chunk keywords
3. Calculate overlap ratio
4. Normalize and weight scores

```python
def getKWOverlapScore(self, query, chunks, percentage_val):
    doc = self.nlp(query)
    query_entities = set([ent.text.lower() for ent in doc.ents])
    # Calculate overlap with chunk keywords
```

### 3.4 Vector Similarity Scoring
**Purpose**: Semantic similarity preservation
**Process**:
1. Use pre-computed chunk embeddings
2. Normalize similarity scores
3. Apply weight to final score

### 3.5 Final Score Calculation
```python
# Initialize final score
chunk['final_score'] = 0

# Apply each scoring method
chunk['final_score'] += tfidf_score * (20/100)
chunk['final_score'] += keyword_score * (30/100)
chunk['final_score'] += vector_score * (50/100)

# Sort by final score
chunks = sorted(chunks, key=lambda x: x['final_score'], reverse=True)
```

---

## 4. Chain of Thoughts Processing

### 4.1 Iterative Self-Improvement

The Chain of Thoughts (CoT) system enables LLMs to iteratively improve their responses through self-reflection and refinement.

#### CoT Workflow
```python
def _chain_of_thoughts_process(self, state: PipelineState):
    current_response = self.master_llm_tool._run(
        operations="initial_response",
        context_text=state.context_text,
        user_query=state.user_query
    )
    
    for iteration in range(state.max_iterations):
        cot_response = self.cot_tool._run(
            context_text=state.context_text,
            user_query=state.user_query,
            previous_response=current_response,
            iteration_number=iteration + 1,
        )
```

### 4.2 Self-Evaluation Criteria
1. **Accuracy**: Factual correctness according to context
2. **Completeness**: Coverage of all relevant information
3. **Clarity**: Explanation quality and structure
4. **Reasoning**: Logical flow and analysis depth
5. **Context Usage**: Full utilization of provided context
6. **Image Integration**: Proper image tag embedding

### 4.3 Improvement Strategies
- **DEEPEN**: Add detailed explanations or nuanced understanding
- **RESTRUCTURE**: Improve organization and logical flow
- **EXPAND**: Include overlooked context elements
- **REFINE**: Enhance clarity, precision, or coherence
- **VALIDATE**: Confirm accuracy and fix errors
- **SYNTHESIZE**: Better connect different information pieces

### 4.4 CoT Output Format
```json
{
  "improved_answer": "Refined answer with image tags",
  "improvement_strategy": "DEEPEN|RESTRUCTURE|EXPAND|REFINE|VALIDATE|SYNTHESIZE",
  "changes_made": ["List of specific improvements"],
  "reasoning": "Why changes improve the answer",
  "confidence_score": 8,
  "continue_iteration": true,
  "iteration_summary": "What this iteration accomplished",
  "next_focus": "What next iteration should focus on",
  "relevant_image_tags": ["hash_1", "hash_2"],
  "quality_progression": "How quality improved from previous iteration"
}
```

---

## 5. Adaptive JSON Extraction

### 5.1 Intelligent Response Parsing

The Adaptive JSON Extractor handles various LLM response formats and ensures consistent data extraction across different models and prompts.

#### Prompt Schema System
```python
self.prompt_schemas = {
    'DirectResponsePrompt': {
        "required": ["answer", "reasoning", "context_coverage", "relevant_image_tags"],
        "optional": []
    },
    'MasterLLMPrompt': {
        "required": ["answer", "reasoning", "confidence_score", "needs_review", "relevant_image_tags", "context_coverage"],
        "optional": ["review_reason"]
    },
    # ... additional schemas
}
```

### 5.2 Extraction Methods

#### Method 1: Brace Matching
- Tracks opening/closing braces
- Handles nested JSON structures
- Robust against malformed JSON

#### Method 2: Regex Pattern Matching
- Uses regex patterns to find JSON blocks
- Handles markdown-formatted JSON
- Extracts from code blocks

#### Method 3: Direct JSON Parsing
- Attempts direct JSON parsing
- Fallback for simple cases
- Error handling for malformed JSON

### 5.3 Schema Validation
```python
def _extract_by_schema(self, parsed_json, prompt_type):
    schema = self.prompt_schemas[prompt_type]
    extracted = {"_prompt_type": prompt_type}
    
    # Extract required fields
    for key in schema['required']:
        extracted[key] = parsed_json.get(key)
    
    # Extract optional fields
    for key in schema.get("optional", []):
        if key in parsed_json:
            extracted[key] = parsed_json[key]
    
    return extracted
```

### 5.4 Automatic Schema Detection
- Analyzes JSON structure against known schemas
- Calculates matching scores
- Selects best-fitting schema
- Handles unknown response formats

---

## 6. Image Integration System

### 6.1 Image Tag System

#### Tag Format
```xml
<image_dec>
  <image_id>hash_123</image_id>
  Description text
</image_dec>
```

#### Integration in Responses
- Images are referenced using `<image_id>hash_123</image_id>` tags
- Tags are embedded directly in answer text
- System ensures image relevance and proper placement

### 6.2 Image Processing Pipeline

#### Step 1: Image Extraction
- Extract images from uploaded documents
- Generate unique image IDs (UUID)
- Store images in vector database

#### Step 2: Image Storage
```python
IMAGE_STORAGE_FOLDER_PATH = 'images/document_images/'
IMAGES_URL = 'document_images'
```

#### Step 3: Image Retrieval
- Retrieve relevant images based on chunk content
- Convert to base64 for web display
- Include in response with proper tags

### 6.3 Image Relevance Scoring
- Analyze chunk content for image references
- Calculate image-chunk semantic similarity
- Filter images by relevance threshold
- Ensure proper image-answer alignment

---

## 7. Document Processing Pipeline

### 7.1 Multi-Format Support

#### Supported Formats
- **PDF Documents**: Text and image extraction
- **Text Files**: `.txt`, `.md`, `.docx`
- **Scanned Documents**: OCR via Mistral API

#### Processing Steps
1. **Document Parsing**: Extract text and images
2. **Chunking**: Semantic chunk creation
3. **Embedding Generation**: Vector representation
4. **Vector Storage**: Database indexing
5. **Image Processing**: Separate image handling

### 7.2 OCR Processing (Mistral API)
```python
class MistralOCR:
    def requestOcrModel(self, base64_pdf):
        # Send PDF to Mistral OCR API
        # Return structured OCR results
```

#### OCR Features
- **Multi-page Processing**: Handle large documents
- **Image Extraction**: Separate images from text
- **Metadata Preservation**: Maintain document structure
- **Language Detection**: Automatic language identification

### 7.3 Batch Processing
```python
UPLOAD_BATCH = 50  # Files processed per batch
```

#### Batch Optimization
- Parallel processing for multiple files
- Memory management for large uploads
- Progress tracking and error handling
- Temporary file cleanup

---

## 8. Vector Database Integration

### 8.1 Supabase Vector Database

#### Database Schema
```sql
-- Chunks table
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768),
    metadata JSONB,
    file_info JSONB,
    chunk_info JSONB,
    semantic_info JSONB,
    media_ref JSONB,
    position_info JSONB
);

-- Images table
CREATE TABLE images (
    id UUID PRIMARY KEY,
    image_url TEXT,
    image_data BYTEA,
    metadata JSONB
);
```

#### Vector Search Functions
```sql
-- Similarity search function
CREATE OR REPLACE FUNCTION similarity_search_chunks(
    query_embedding VECTOR(768),
    match_threshold FLOAT,
    match_count INT,
    file_filter JSONB,
    chunk_filter JSONB,
    language_filter TEXT,
    chunk_type_filter TEXT
)
```

### 8.2 Search Configuration
```python
TOP_K = 7                    # Number of chunks to retrieve
SIMILARITY_TH = 0.5          # Similarity threshold
TOTAL_CHUNKS_CONSIDERED = 5  # Chunks for processing
TOTAL_CHUNKS_COVERAGE = 80   # Coverage percentage
```

### 8.3 Advanced Filtering
- **File Filtering**: Search within specific documents
- **Language Filtering**: Multi-language support
- **Chunk Type Filtering**: Different chunk types
- **Semantic Filtering**: Keyword-based filtering

---

## 9. Response Generation Modes

### 9.1 Single LLM Mode
**Use Case**: Simple, straightforward queries
**Process**: Direct response generation
**Benefits**: Fast processing, low resource usage
**Configuration**: Single model selection

### 9.2 Multi-LLM Jury Mode
**Use Case**: Complex queries requiring multiple perspectives
**Process**: Ensemble deliberation with master coordination
**Benefits**: Higher accuracy, error correction
**Configuration**: Multiple models, iteration control

### 9.3 Chain of Thoughts Mode
**Use Case**: Analytical queries requiring deep reasoning
**Process**: Iterative self-improvement
**Benefits**: Enhanced reasoning, detailed explanations
**Configuration**: Iteration limits, improvement strategies

### 9.4 Model Adaptations
```python
ADAPTATIONS = {
    'gpt-4o': {
        'style': 'comprehensive',
        'complexity': 'high',
        'json_strict': True
    },
    'gpt-3.5-turbo': {
        'style': 'concise',
        'complexity': 'moderate',
        'json_strict': True
    }
    # ... additional model configurations
}
```

---

## 10. Advanced Configuration Options

### 10.1 Token Management
```python
MAX_ALLOWED_TOKENS = {
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000,
    'Claude 2': 10000,
    'Claude Instant 2': 10000
}
```

### 10.2 Temperature and Creativity Control
```python
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THINKING_ITERATIONS = 3
```

### 10.3 Entity Filtering
```python
ENTITIES_TO_IGNORE = ["\\in", "###", "$", "|", "$ | $"]
```

### 10.4 Performance Optimization
- **Batch Size Control**: Configurable upload batches
- **Memory Management**: Efficient resource usage
- **Parallel Processing**: Multi-threaded operations
- **Caching**: Response and embedding caching

---

## Summary

The RAG-Enhanced Chatbot represents a sophisticated AI system with multiple advanced features:

1. **Multi-LLM Jury System**: Ensemble approach for improved accuracy
2. **Advanced Chunking**: Semantic-aware document segmentation
3. **Intelligent Re-ranking**: Multi-factor relevance scoring
4. **Chain of Thoughts**: Iterative self-improvement
5. **Adaptive JSON Extraction**: Robust response parsing
6. **Image Integration**: Visual context support
7. **Comprehensive Pipeline**: End-to-end document processing
8. **Vector Database**: Scalable similarity search
9. **Multiple Generation Modes**: Flexible response creation
10. **Advanced Configuration**: Fine-grained system control

This system provides enterprise-grade document processing and question-answering capabilities with state-of-the-art AI techniques and robust error handling. 