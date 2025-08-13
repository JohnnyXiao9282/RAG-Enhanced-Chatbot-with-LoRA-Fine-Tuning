# RAG-Enhanced Chatbot User Manual

## Table of Contents
1. [Getting Started](#1-getting-started)
2. [Using the Application](#2-using-the-application)
3. [Troubleshooting](#3-troubleshooting)
4. [Tips and Best Practices](#4-tips-and-best-practices)

---

## 1. Getting Started

### 1.1 System Requirements

#### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended for optimal performance
- **Storage**: At least 10GB free space for models and vector database
- **GPU**: CUDA-compatible GPU (NVIDIA) recommended for LoRA fine-tuning
  - Minimum: 4GB VRAM
  - Recommended: 8GB+ VRAM for large models

#### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Node.js**: Version 16 or higher (for frontend)
- **Git**: For cloning the repository

#### Network Requirements
- Stable internet connection for:
  - Downloading model weights
  - API calls to external LLM services
  - Vector database operations

### 1.2 Installation Guide

#### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/RAG-Enhanced-Chatbot-with-LoRA-Fine-Tuning.git
cd RAG-Enhanced-Chatbot-with-LoRA-Fine-Tuning
```

#### Step 2: Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the backend directory with the following variables:
   ```env
   # API Keys
   OPEN_AI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GEMINI_API_KEY=your_gemini_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   MISTRAL_API_KEY=your_mistral_api_key
   
   # Vector Database
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   
   # Model Configuration
   ENCODER_NAME=sentence_encoder
   ENCODING_MODEL=all-mpnet-base-v2
   PARSER=MistralOCR
   CHUNKER=groupEmbeddingAlgo
   ```

5. **Install additional system dependencies (if needed):**
   ```bash
   # For OCR functionality
   sudo apt-get install tesseract-ocr  # Ubuntu/Debian
   brew install tesseract              # macOS
   ```

#### Step 3: Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../frontend/electric-rag
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

#### Step 4: Start the Application

1. **Start the backend server:**
   ```bash
   # From the backend directory
   python main.py
   ```
   The backend will start on `http://localhost:8000`

2. **Start the frontend development server:**
   ```bash
   # From the frontend/electric-rag directory
   npm start
   ```
   The frontend will start on `http://localhost:3000`

3. **Access the application:**
   Open your browser and navigate to `http://localhost:3000`

---

## 2. Using the Application

### 2.1 Uploading Files

#### Supported File Formats
- **PDF Documents**: Standard PDF files with text and images
- **Text Files**: `.txt`, `.md`, `.docx` files
- **Scanned Documents**: PDFs with OCR processing via Mistral API

#### Upload Process

1. **Access the Upload Panel:**
   - Click on the "Upload" tab in the left sidebar
   - Or use the upload icon in the main interface

2. **Select Files:**
   - Click "Choose Files" or drag and drop files into the upload area
   - You can select multiple files at once
   - Maximum file size: 50MB per file

3. **Configure Upload Settings:**
   - **Train Model**: Enable this option if you want to fine-tune the model with your documents
   - **Batch Processing**: Files are processed in batches of 50 for optimal performance

4. **Monitor Upload Progress:**
   - The interface shows upload progress for each file
   - Processing status is displayed in real-time
   - Success/failure indicators for each file

#### File Processing Pipeline

1. **Document Parsing**: Files are parsed using the configured parser (default: MistralOCR)
2. **Chunking**: Documents are split into manageable chunks (default: 1000 tokens with 50 token overlap)
3. **Embedding Generation**: Text chunks are converted to vector embeddings
4. **Vector Storage**: Embeddings are stored in the vector database (Supabase)
5. **Image Processing**: Images are extracted and stored separately

### 2.2 Performing Query Searches

#### Basic Search

1. **Enter Your Query:**
   - Type your question in the chat input field
   - Press Enter or click the Send button

2. **Search Configuration:**
   - **Number of Neighbors**: Set how many related chunks to retrieve (default: 7)
   - **Similarity Threshold**: Minimum similarity score for relevant chunks (default: 0.5)
   - **Fetch Chains**: Enable to retrieve connected chunks for better context

#### Advanced Search Options

1. **Multi-LLM Processing:**
   - Enable "Multi-LLM" for jury-based response generation
   - Select multiple LLM models to participate in the decision process
   - Choose an expert LLM for final response generation

2. **Chain of Thought:**
   - Enable for step-by-step reasoning
   - Useful for complex queries requiring detailed analysis

3. **Model Selection:**
   - Choose from available models: GPT-3.5, GPT-4, Claude, Gemini, etc.
   - Configure temperature settings for response creativity

#### Search Results

The system returns:
- **Primary Answer**: The main response to your query
- **Reasoning**: Step-by-step explanation of how the answer was derived
- **Relevant Images**: Any images from the documents that support the answer
- **Source Chunks**: The document sections used to generate the response

### 2.3 Managing Chunks and Re-ranking

#### Chunk Management

1. **Chunk Configuration:**
   - **Chunk Length**: Adjust the size of document chunks (default: 1000 tokens)
   - **Overlap**: Set overlap between chunks for better context (default: 50 tokens)
   - **Chunk Type**: Choose between sentence-based or page-based chunking

2. **Re-ranking Parameters:**
   - **TF-IDF Weight**: 20% - Term frequency-inverse document frequency scoring
   - **Keyword Weight**: 30% - Keyword-based relevance scoring
   - **Vector Similarity Weight**: 50% - Semantic similarity scoring

#### Chunk Retrieval Options

1. **Top-K Retrieval:**
   - Retrieve the top K most similar chunks
   - Default: 7 chunks
   - Adjustable based on query complexity

2. **Similarity Threshold:**
   - Filter chunks below a certain similarity score
   - Default: 0.5
   - Higher values ensure more relevant results

3. **File Filtering:**
   - Filter chunks by specific documents
   - Useful for domain-specific queries

### 2.4 Generating Responses

#### Response Generation Modes

1. **Single LLM Mode:**
   - Uses one selected model for response generation
   - Fastest processing time
   - Suitable for straightforward queries

2. **Multi-LLM Jury Mode:**
   - Multiple models evaluate and vote on responses
   - Higher accuracy and reliability
   - Slower processing time
   - Configurable iterations (default: 3)

3. **Chain of Thought Mode:**
   - Step-by-step reasoning process
   - Detailed explanation of the thinking process
   - Best for complex analytical queries

#### Response Quality Features

1. **Context Coverage:**
   - System ensures responses are grounded in uploaded documents
   - Provides confidence scores for response accuracy
   - Highlights relevant document sections

2. **Image Integration:**
   - Automatically includes relevant images from documents
   - Supports visual context in responses
   - Images are base64 encoded for web display

3. **Response Validation:**
   - JSON-structured responses for consistency
   - Error handling for malformed responses
   - Fallback mechanisms for failed generations

---

## 3. Troubleshooting

### 3.1 Common Issues

#### Installation Issues

**Problem**: Python dependencies fail to install
- **Solution**: Ensure you're using Python 3.8+ and have pip updated
- **Solution**: Try installing dependencies one by one to identify conflicts

**Problem**: CUDA/GPU issues
- **Solution**: Install appropriate CUDA drivers for your GPU
- **Solution**: Use CPU-only mode if GPU is not available

**Problem**: Node.js/npm issues
- **Solution**: Update Node.js to version 16 or higher
- **Solution**: Clear npm cache: `npm cache clean --force`

#### Runtime Issues

**Problem**: Backend server won't start
- **Solution**: Check if port 8000 is available
- **Solution**: Verify all environment variables are set correctly
- **Solution**: Check logs in `backend/logs/` directory

**Problem**: Frontend can't connect to backend
- **Solution**: Ensure backend is running on `http://localhost:8000`
- **Solution**: Check CORS settings in backend configuration
- **Solution**: Verify proxy settings in `package.json`

**Problem**: File upload fails
- **Solution**: Check file size limits (50MB per file)
- **Solution**: Verify file format is supported
- **Solution**: Ensure sufficient disk space for temporary files

**Problem**: API key errors
- **Solution**: Verify all API keys are valid and have sufficient credits
- **Solution**: Check rate limits for external services
- **Solution**: Ensure keys are properly formatted in `.env` file

#### Performance Issues

**Problem**: Slow response times
- **Solution**: Reduce number of chunks retrieved
- **Solution**: Use single LLM mode instead of multi-LLM
- **Solution**: Optimize chunk size and overlap settings

**Problem**: High memory usage
- **Solution**: Reduce batch size for file processing
- **Solution**: Use smaller embedding models
- **Solution**: Implement chunk cleanup procedures

**Problem**: Vector database connection issues
- **Solution**: Check Supabase credentials and connection
- **Solution**: Verify database schema is properly set up
- **Solution**: Monitor database usage limits

### 3.2 Contact Support

#### Getting Help

1. **Check Documentation:**
   - Review this user manual thoroughly
   - Check the main README.md file
   - Examine code comments for configuration options

2. **Log Analysis:**
   - Check backend logs in `backend/logs/`
   - Monitor browser console for frontend errors
   - Review API response logs

3. **Community Support:**
   - Create an issue on the GitHub repository
   - Include detailed error messages and system information
   - Provide steps to reproduce the problem

4. **Contact Information:**
   - **GitHub Issues**: [Repository Issues Page]
   - **Email**: [Maintainer Email]
   - **Documentation**: [Project Wiki]

#### Bug Reporting

When reporting issues, include:
- **System Information**: OS, Python version, Node.js version
- **Error Messages**: Complete error logs and stack traces
- **Steps to Reproduce**: Detailed steps to recreate the issue
- **Expected vs Actual Behavior**: Clear description of what should happen
- **Screenshots**: Visual evidence of the problem (if applicable)

---

## 4. Tips and Best Practices

### 4.1 Document Preparation

#### Optimal Document Structure
- **Use clear headings and subheadings** for better chunk organization
- **Include relevant images** with descriptive captions
- **Maintain consistent formatting** across documents
- **Avoid very large single documents** - split into logical sections

#### File Organization
- **Group related documents** in the same upload session
- **Use descriptive filenames** for easier identification
- **Maintain document version control** for updates
- **Archive old documents** to prevent confusion

### 4.2 Query Optimization

#### Effective Query Techniques
- **Be specific and detailed** in your questions
- **Use relevant keywords** from your documents
- **Ask follow-up questions** for clarification
- **Reference specific documents** when possible

#### Query Types
- **Factual Questions**: "What is the main function of mitochondria?"
- **Comparative Questions**: "How does process A differ from process B?"
- **Analytical Questions**: "What are the implications of this finding?"
- **Procedural Questions**: "What are the steps to complete this task?"

### 4.3 System Configuration

#### Performance Optimization
- **Adjust chunk size** based on document complexity
- **Optimize similarity thresholds** for your use case
- **Use appropriate LLM models** for different query types
- **Monitor resource usage** and adjust accordingly

#### Security Best Practices
- **Secure API keys** and never commit them to version control
- **Use environment variables** for sensitive configuration
- **Implement proper access controls** for production deployments
- **Regularly update dependencies** for security patches

### 4.4 Maintenance

#### Regular Tasks
- **Monitor system performance** and adjust settings as needed
- **Clean up temporary files** to free disk space
- **Update model weights** and dependencies regularly
- **Backup vector database** and important configurations

#### Data Management
- **Archive old documents** to maintain system performance
- **Review and update** document relevance periodically
- **Monitor embedding quality** and retrain if necessary
- **Track usage patterns** to optimize system configuration

### 4.5 Advanced Features

#### Custom Model Training
- **Prepare training data** with high-quality examples
- **Use appropriate fine-tuning** parameters for your domain
- **Validate model performance** before deployment
- **Monitor model drift** and retrain as needed

#### Integration Capabilities
- **API Integration**: Use the REST API for custom applications
- **Webhook Support**: Set up notifications for processing events
- **Batch Processing**: Handle large document collections efficiently
- **Multi-tenant Support**: Configure for multiple user organizations

---

## Conclusion

This RAG-Enhanced Chatbot provides a powerful platform for document-based question answering with advanced AI capabilities. By following this user manual, you can effectively set up, configure, and use the system to extract valuable insights from your document collections.

For additional support or feature requests, please refer to the contact information provided in the troubleshooting section. 