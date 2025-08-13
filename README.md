# Electric RAG - Advanced Retrieval-Augmented Generation System

## 📌 Overview

**Electric RAG** is a sophisticated Retrieval-Augmented Generation (RAG) system that combines document processing, semantic search, and multi-LLM orchestration to provide intelligent, context-aware responses. The system features a FastAPI backend with advanced document ingestion, embedding generation, and a React frontend for seamless user interaction.

## UI
# Chat UI
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/d6aba562-ad12-4492-9672-07d42c2bb119" />


# Document Ingestion
<img width="1919" height="1078" alt="image" src="https://github.com/user-attachments/assets/edb0ec8d-9d29-49f1-b64d-4d18e4e88f48" />


---

## 🚀 Key Features

- 🔍 **Advanced RAG Pipeline:** Complete document ingestion, chunking, embedding, and retrieval system
- 🧠 **Multi-LLM Orchestration:** Support for multiple language models with jury-based decision making
- 📄 **Multi-Format Document Processing:** PDF parsing with OCR support, text extraction, and JSON storage
- 🎯 **Intelligent Chunking:** Context-aware document segmentation with configurable overlap
- 🔗 **Chain of Thought Processing:** Enhanced reasoning capabilities with iterative thinking
- 🖼️ **Image-Aware Processing:** OCR-based image extraction and multi-modal content handling
- 📊 **Vector Database Integration:** Supabase vector database with analytics and monitoring
- 🌐 **Modern Web Interface:** React frontend with Tailwind CSS for optimal user experience
- ⚡ **FastAPI Backend:** High-performance API with CORS support and comprehensive endpoints

---

## 🛠️ Tech Stack

### Backend
- **Framework:** FastAPI with Uvicorn
- **Language Models:** GPT-4, Claude-3, Gemini-Pro, Llama-2, Mistral-7B
- **Embeddings:** Sentence Transformers (all-mpnet-base-v2)
- **Vector Database:** Supabase Vector Database
- **Document Processing:** PyMuPDF, Mistral OCR
- **Chunking:** Advanced sentence and page-based algorithms
- **Reranking:** TF-IDF, keyword matching, and vector similarity

### Frontend
- **Framework:** React 19 with modern hooks
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **Build Tool:** Create React App

---

## 📂 Project Structure

```
RAG/
├── backend/                          # FastAPI backend application
│   ├── chunking/                     # Document chunking algorithms
│   │   ├── chunker.py               # Main chunking interface
│   │   └── chunkers/                # Specific chunking implementations
│   ├── config/                       # Configuration settings
│   │   └── settings.py              # Global configuration parameters
│   ├── data/                         # Data storage and processing
│   │   ├── raw_document_data/       # Temporary document storage
│   │   └── raw_jsons/               # Processed JSON outputs
│   ├── database/                     # Database layer
│   │   ├── relational/               # Relational database components
│   │   └── vector/                   # Vector database integration
│   │       ├── vectorDB.py          # Vector database interface
│   │       └── vectorDBs/           # Specific vector DB implementations
│   ├── embedding/                    # Embedding generation
│   │   ├── encoder.py               # Main encoder interface
│   │   └── encoders/                # Specific encoder implementations
│   ├── generation/                   # LLM generation components
│   ├── helpers/                      # Utility helper modules
│   ├── ingestion/                    # Document ingestion pipeline
│   │   └── ingestion.py             # Main ingestion orchestrator
│   ├── llmservice/                   # LLM service layer
│   │   ├── multillmorchestrator.py  # Multi-LLM orchestration
│   │   ├── adaptiveJsonExtractor.py # JSON response extraction
│   │   ├── llmmodels.py             # LLM model definitions
│   │   └── prompts.py               # System prompts
│   ├── parsing/                      # Document parsing
│   │   ├── parser.py                # Main parser interface
│   │   └── parsers/                 # Specific parser implementations
│   ├── retriever/                    # Retrieval system
│   │   ├── retriever.py             # Main retrieval logic
│   │   └── reranker/                # Reranking algorithms
│   ├── utils/                        # Utility functions
│   │   └── logger.py                # Logging configuration
│   └── main.py                      # FastAPI application entry point
├── frontend/                         # React frontend application
│   └── electric-rag/                # Main frontend package
│       ├── src/                     # React source code
│       ├── public/                  # Static assets
│       └── package.json             # Frontend dependencies
├── requirements.txt                  # Python dependencies
└── README.md                        # Project documentation
```

---

## ✅ Core Functionality

### Document Processing
- **Multi-format Support:** PDF, text, and scanned documents
- **OCR Integration:** Mistral OCR for image-based content extraction
- **Smart Chunking:** Configurable chunk sizes with overlap optimization
- **Content Preservation:** Maintains document structure and formatting

### Retrieval System
- **Semantic Search:** Vector similarity-based document retrieval
- **Multi-modal Retrieval:** Text and image content matching
- **Reranking:** Combines TF-IDF, keyword matching, and vector similarity
- **Configurable Top-K:** Adjustable number of retrieved chunks

### LLM Orchestration
- **Multi-LLM Support:** Integration with multiple language model providers
- **Jury-based Decisions:** Consensus-based response generation
- **Chain of Thought:** Iterative reasoning for complex queries
- **Expert LLM Selection:** Specialized model selection for specific tasks

### API Endpoints
- **Chat Interface:** `/api/chat` - Main conversation endpoint
- **Document Upload:** `/api/upload` - File ingestion with optional training
- **Analytics:** `/api/analytics` - Vector database statistics
- **Health Check:** `/api/health` - System status monitoring
- **LLM Models:** `/api/llm-models` - Available model information

---

## 🧪 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+ (for frontend)
- Supabase account (for vector database)
- API keys for LLM providers

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAG/backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file with:
   ```env
   ENCODER_NAME=sentence_encoder
   ENCODING_MODEL=all-mpnet-base-v2
   PARSER=mistral_ocr
   CHUNKER=group_embedding
   ```

4. **Start the backend server:**
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../frontend/electric-rag
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   The frontend will be available at `http://localhost:3000`

---

## 🔧 Configuration

### Backend Settings (`backend/config/settings.py`)
- **Chunking:** `DEFAULT_CHUNK_LENGTH`, `DEFAULT_OVER_LAP`
- **Retrieval:** `TOP_K`, `SIMILARITY_TH`
- **Processing:** `DEFAULT_TEMPERATURE`, `DEFAULT_THINKING_ITERATIONS`
- **Storage:** `TEMP_FILES_DOC`, `TEMP_FILES_JSONS`

### Vector Database
- Configure Supabase connection in `backend/database/vector/vectorDBs/supabasevdb.py`
- Set up appropriate indexes for optimal search performance

### LLM Providers
- Configure API keys for supported providers
- Adjust model selection in `backend/llmservice/llmmodels.py`

---

## 📊 API Usage

### Chat Request
```json
{
  "query": "What is the main topic of the document?",
  "multiLLM": false,
  "fetchChains": true,
  "noOfNeighbours": 5,
  "activeLLMs": [],
  "expertLLM": "GPT-4",
  "chainOfThought": true
}
```

### Document Upload
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@document.pdf" \
  -F "train_model=false"
```

---

## 🚀 Deployment

### Production Considerations
- Use proper environment variable management
- Implement authentication and authorization
- Set up monitoring and logging
- Configure CORS origins for production domains
- Use production-grade vector database instances

### Docker Support
- Backend can be containerized with FastAPI
- Frontend can be built and served statically
- Use environment-specific configuration files

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Add comprehensive tests for new features
- Update documentation for API changes

---

## 📞 Support

- **Issues:** Create GitHub issues for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Documentation:** Check the codebase for implementation details

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🔮 Roadmap

- [ ] Enhanced multi-modal support
- [ ] Advanced reranking algorithms
- [ ] Real-time collaboration features
- [ ] Mobile application
- [ ] Advanced analytics dashboard
- [ ] Plugin system for custom processors
