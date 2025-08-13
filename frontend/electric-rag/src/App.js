import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageCircle, 
  Upload, 
  Database, 
  Send, 
  X, 
  FileText, 
  CheckCircle, 
  Loader, 
  Zap, 
  Brain, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  Eye, 
  EyeOff,
  Link,
  Hash,
  Lightbulb,
  Settings
} from 'lucide-react';

const App = () => {
  const [activePanel, setActivePanel] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [trainModel, setTrainModel] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState({});
  
  // Enhanced LLM Configuration States
  const [useMultiLLM, setUseMultiLLM] = useState(false);
  const [activeLLMs, setActiveLLMs] = useState([]);
  const [expertLLM, setExpertLLM] = useState('gpt-4o');
  const [fetchChains, setFetchChains] = useState(false);
  const [noOfNeighbours, setNoOfNeighbours] = useState(0);
  const [chainOfThought, setChainOfThought] = useState(false);
  
  // Analytics States
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsError, setAnalyticsError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const availableLLMs = [
    'gpt-3',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-4o',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4-32k',
    'Claude 2',
    'Claude Instant 2',
    'Claude 3',
    'Gemini-Pro',
    'Llama-2',
    'Mistral-7B'
  ];

  const expertLLMOptions = [
    'gpt-3',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-4o',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4-32k',
    'Claude 2',
    'Claude Instant 2',
    'Claude 3',
    'Gemini-Pro',
    'Llama-2',
    'Mistral-7B'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load analytics data when analytics panel is active
  useEffect(() => {
    if (activePanel === 'analytics') {
      loadAnalyticsData();
    }
  }, [activePanel]);

  const loadAnalyticsData = async () => {
    setAnalyticsLoading(true);
    setAnalyticsError(null);
    
    try {
      const [analyticsResponse, simpleResponse] = await Promise.all([
        fetch('/api/analytics'),
        fetch('/api/analytics/simple')
      ]);

      if (!analyticsResponse.ok || !simpleResponse.ok) {
        throw new Error('Failed to fetch analytics data');
      }

      const [fullAnalytics, simpleStats] = await Promise.all([
        analyticsResponse.json(),
        simpleResponse.json()
      ]);
      setAnalyticsData({
        ...fullAnalytics,
        simple: simpleStats
      });
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error loading analytics:', error);
      setAnalyticsError(error.message);
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const toggleReasoning = (messageIndex) => {
    setExpandedReasoning(prev => ({
      ...prev,
      [messageIndex]: !prev[messageIndex]
    }));
  };

  const handleLLMToggle = (llm) => {
    setActiveLLMs(prev => 
      prev.includes(llm) 
        ? prev.filter(l => l !== llm)
        : [...prev, llm]
    );
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { role: 'user', content: inputMessage };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Fixed payload with explicit field names and debugging
      const payload = {
        query: inputMessage,
        multiLLM: useMultiLLM,
        activeLLMs: useMultiLLM ? activeLLMs : [],
        expertLLM: useMultiLLM ? expertLLM : "gpt-4o",
        fetchChains: fetchChains,
        noOfNeighbours: noOfNeighbours,
        chainOfThought: chainOfThought
      };

      // Debug logging
      console.log("Frontend State Values:");
      console.log("- chainOfThought state:", chainOfThought);
      console.log("- useMultiLLM state:", useMultiLLM);
      console.log("- fetchChains state:", fetchChains);
      console.log("Sending payload:", JSON.stringify(payload, null, 2));

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if(!response.ok){
        const errorText = await response.text();
        console.error("Error response: ", errorText);
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      
      // Process images in the response
      let processedContent = data.response;
      let hasImages = false;

      if (data.images && Object.keys(data.images).length > 0) {
        
        const imageTagRegex = /<image_id>(.*?)<\/image_id>/g;

        processedContent = processedContent.replace(imageTagRegex, (match, imageId) => {
          if (data.images[imageId]){
            hasImages = true;
            return `<img src="data:image/png;base64,${data.images[imageId]}" alt="Retrieved Image" style="max-width: 100%; height: auto; border-radius: 8px; margin: 8px 0; border: 1px solid #e5e7eb; display: block;" />`;
          } else {
            console.warn(`Image with ID "${imageId}" not found in response`);
            return `<div style="padding: 8px; margin: 8px 0; background-color: #f3f4f6; border: 1px solid #d1d5db; border-radius: 8px; color: #6b7280; text-align: center;">Image not found: ${imageId}</div>`;
          }
        });
      }

      const assistantMessage = { 
        role: 'assistant', 
        content: processedContent,
        reasoning: data.reasoning || '',
        isHtml: hasImages
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: `Sorry, there was an error processing your request: ${error.message}`,
        reasoning: '' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    setUploadedFiles(files);
    setIsUploading(true);

    try {
      const formData = new FormData();
      files.forEach(file => {
        formData.append('files', file);
      });
      formData.append('train_model', trainModel);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const success = await response.json();
      
      if (success) {
        alert('Files uploaded successfully!');
        setUploadedFiles([]);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      } else {
        alert('Upload failed. Please try again.');
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleToggleMultiModel = () => {
    setUseMultiLLM(prev => {
      if (!prev) setChainOfThought(false);  // Disable other option if enabling this one
      return !prev;
    });
  };
  
  const handleToggleChainOfThought = () => {
    setChainOfThought(prev => {
      if (!prev) setUseMultiLLM(false); // Disable other option if enabling this one
      return !prev;
    });
  };

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const renderSidePanel = () => (
    <div className="w-80 bg-white text-gray-800 p-6 flex flex-col h-full border-r border-gray-200 relative overflow-hidden shadow-lg">
      {/* Subtle accent line */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"></div>
      
      <div className="relative z-10 overflow-y-auto">
        <h1 className="text-2xl font-bold mb-8 text-gray-800 tracking-tight">
          AI Assistant Pro
        </h1>
        
        {/* Panel Options */}
        <div className="space-y-3 mb-8">
          <button
            onClick={() => setActivePanel('chat')}
            className={`w-full group relative overflow-hidden p-4 rounded-lg transition-all duration-300 ${
              activePanel === 'chat' 
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md' 
                : 'bg-gray-50 hover:bg-gray-100 border border-gray-200 hover:border-blue-300 text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <MessageCircle size={20} className={activePanel === 'chat' ? 'text-white' : 'text-blue-600'} />
              <span className="font-semibold">Conversation</span>
            </div>
          </button>
          
          <button
            onClick={() => setActivePanel('upload')}
            className={`w-full group relative overflow-hidden p-4 rounded-lg transition-all duration-300 ${
              activePanel === 'upload' 
                ? 'bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-md' 
                : 'bg-gray-50 hover:bg-gray-100 border border-gray-200 hover:border-green-300 text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <Upload size={20} className={activePanel === 'upload' ? 'text-white' : 'text-green-600'} />
              <span className="font-semibold">Document Upload</span>
            </div>
          </button>
          
          <button
            onClick={() => setActivePanel('analytics')}
            className={`w-full group relative overflow-hidden p-4 rounded-lg transition-all duration-300 ${
              activePanel === 'analytics' 
                ? 'bg-gradient-to-r from-purple-600 to-violet-600 text-white shadow-md' 
                : 'bg-gray-50 hover:bg-gray-100 border border-gray-200 hover:border-purple-300 text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-3 relative z-10">
              <Database size={20} className={activePanel === 'analytics' ? 'text-white' : 'text-purple-600'} />
              <span className="font-semibold">Analytics</span>
            </div>
          </button>
        </div>

        {/* Enhanced LLM Configuration Section */}
        <div className="space-y-4 border-t border-gray-200 pt-6">
          <div className="flex items-center space-x-2 mb-4">
            <Settings size={18} className="text-gray-600" />
            <h3 className="text-gray-700 font-semibold text-sm tracking-wide">Configuration</h3>
          </div>

          {/* Multi-LLM Toggle */}
          <div className="flex items-center space-x-3 bg-blue-50 p-3 rounded-lg border border-blue-200">
            <div className="relative">
              <input
                type="checkbox"
                id="multiLLM"
                checked={useMultiLLM}
                onChange={(e) => {
                  console.log("Multi-LLM toggled:", e.target.checked);
                  setUseMultiLLM(e.target.checked);
                }}
                className="w-5 h-5 text-blue-600 bg-white border-2 border-blue-300 rounded focus:ring-blue-500 focus:ring-2"
              />
            </div>
            <label htmlFor="multiLLM" className="text-blue-800 font-medium text-sm">
              Multi-Model Mode
            </label>
          </div>

          {/* Active LLMs Selection */}
          {useMultiLLM && (
            <div className="space-y-3 animate-fadeIn">
              <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                <label className="block text-gray-700 font-medium mb-2 text-xs">
                  Active Models
                </label>
                <div className="grid grid-cols-1 gap-1 max-h-32 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100">
                  {availableLLMs.map(llm => (
                    <div key={llm} className="flex items-center space-x-2 p-2 bg-white rounded border border-gray-200 hover:border-blue-300 transition-all">
                      <input
                        type="checkbox"
                        id={llm}
                        checked={activeLLMs.includes(llm)}
                        onChange={() => handleLLMToggle(llm)}
                        className="w-3 h-3 text-blue-600 bg-white border-2 border-gray-300 rounded focus:ring-blue-500 focus:ring-1"
                      />
                      <label htmlFor={llm} className="text-gray-700 text-xs font-medium flex-1 cursor-pointer">
                        {llm}
                      </label>
                      {activeLLMs.includes(llm) && <Brain size={12} className="text-blue-500" />}
                    </div>
                  ))}
                </div>
              </div>

              {/* Expert LLM Selection */}
              <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                <label className="block text-gray-700 font-medium mb-2 text-xs">
                  Primary Model
                </label>
                <select
                  value={expertLLM}
                  onChange={(e) => setExpertLLM(e.target.value)}
                  className="w-full p-2 bg-white border-2 border-gray-300 rounded-lg text-gray-700 text-xs focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {expertLLMOptions.map(option => (
                    <option key={option} value={option} className="bg-white">{option}</option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* Fetch Chains Toggle */}
          <div className="flex items-center space-x-3 bg-green-50 p-3 rounded-lg border border-green-200">
            <div className="relative">
              <input
                type="checkbox"
                id="fetchChains"
                checked={fetchChains}
                onChange={(e) => {
                  console.log("Fetch Chains toggled:", e.target.checked);
                  setFetchChains(e.target.checked);
                }}
                className="w-4 h-4 text-green-600 bg-white border-2 border-green-300 rounded focus:ring-green-500 focus:ring-2"
              />
            </div>
            <label htmlFor="fetchChains" className="text-green-800 font-medium text-xs">
              Chain Processing
            </label>
          </div>

          {/* Number of Neighbours */}
          <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
            <label className="block text-gray-700 font-medium mb-2 text-xs">
              <Hash size={12} className="inline mr-1" />
              Neighbor Count
            </label>
            <input
              type="number"
              min="0"
              max="10"
              value={noOfNeighbours}
              onChange={(e) => {
                const value = parseInt(e.target.value) || 0;
                console.log("Neural Neighbours changed:", value);
                setNoOfNeighbours(value);
              }}
              className="w-full p-2 bg-white border-2 border-gray-300 rounded-lg text-gray-700 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="0-10"
            />
          </div>

          {/* Chain of Thought Toggle */}
          <div className="flex items-center space-x-3 bg-orange-50 p-3 rounded-lg border border-orange-200">
            <div className="relative">
              <input
                type="checkbox"
                id="chainOfThought"
                checked={chainOfThought}
                onChange={(e) => {
                  console.log("Chain of Thought toggled:", e.target.checked);
                  setChainOfThought(e.target.checked);
                }}
                className="w-4 h-4 text-orange-600 bg-white border-2 border-orange-300 rounded focus:ring-orange-500 focus:ring-2"
              />
            </div>
            <label htmlFor="chainOfThought" className="text-orange-800 font-medium text-xs">
              Reasoning Mode
            </label>
          </div>

          {/* Configuration Summary */}
          <div className="bg-gray-100 p-3 rounded-lg border border-gray-300 mt-4">
            <div className="text-gray-600 text-xs space-y-1">
              <div className="flex justify-between">
                <span>Multi-Model:</span>
                <span className={useMultiLLM ? 'text-green-600 font-medium' : 'text-red-500'}>
                  {useMultiLLM ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Active Models:</span>
                <span className="text-blue-600 font-medium">{activeLLMs.length}</span>
              </div>
              <div className="flex justify-between">
                <span>Chains:</span>
                <span className={fetchChains ? 'text-green-600 font-medium' : 'text-red-500'}>
                  {fetchChains ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Reasoning:</span>
                <span className={chainOfThought ? 'text-green-600 font-medium' : 'text-red-500'}>
                  {chainOfThought ? 'Enabled' : 'Disabled'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderChatPanel = () => (
    <div className="flex-1 flex flex-col h-full bg-gray-50 relative overflow-hidden">
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6 relative z-10">
        {messages.length === 0 ? (
          <div className="text-center mt-32 animate-fadeIn">
            <div className="relative mb-8">
              <MessageCircle size={80} className="mx-auto text-blue-500" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              Welcome to AI Assistant Pro
            </h2>
            <p className="text-gray-600 text-lg">Start a conversation to get intelligent assistance</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-slideIn`}>
              <div className={`max-w-[75%] relative ${
                message.role === 'user' ? 'space-y-0' : 'space-y-3'
              }`}>
                {/* Main message content */}
                <div className={`p-4 rounded-lg relative overflow-hidden ${
                  message.role === 'user' 
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md' 
                    : 'bg-white text-gray-800 border border-gray-200 shadow-sm'
                }`}>
                  {message.isHtml ? (
                    <div 
                      dangerouslySetInnerHTML={{ __html: message.content }}
                      className="prose prose-gray max-w-none"
                    />
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                  )}
                </div>

                {/* Reasoning section - only for assistant messages */}
                {message.role === 'assistant' && message.reasoning && (
                  <div className="space-y-2">
                    {/* Reasoning toggle button */}
                    <button
                      onClick={() => toggleReasoning(index)}
                      className="flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded-lg transition-all duration-200 text-gray-700 hover:text-blue-700 text-sm"
                    >
                      <Brain size={16} className="text-blue-500" />
                      <span>View Reasoning</span>
                      <div className="flex items-center space-x-1 text-xs text-gray-500">
                        {expandedReasoning[index] ? <EyeOff size={14} /> : <Eye size={14} />}
                        {expandedReasoning[index] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </div>
                    </button>

                    {/* Expandable reasoning content */}
                    <div className={`transition-all duration-300 overflow-hidden ${
                      expandedReasoning[index] ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                    }`}>
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex items-center space-x-2 mb-3">
                          <Brain size={18} className="text-blue-600" />
                          <h4 className="text-blue-800 font-medium">Reasoning Process</h4>
                        </div>
                        <div className="text-gray-700 text-sm leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto custom-scrollbar">
                          {message.reasoning}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex justify-start animate-fadeIn">
            <div className="bg-white text-gray-800 p-4 rounded-lg flex items-center space-x-3 border border-gray-200 shadow-sm">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
              <span className="text-blue-700 font-medium">Processing your request...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="p-6 border-t border-gray-200 bg-white">
        <div className="flex space-x-4">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message here..."
            className="flex-1 p-4 bg-white border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800 placeholder-gray-500 transition-all duration-300"
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-md relative overflow-hidden group"
          >
            <Send size={20} className="relative z-10" />
          </button>
        </div>
      </div>
    </div>
  );

  const renderUploadPanel = () => (
    <div className="flex-1 p-8 bg-gray-50 relative overflow-hidden">

      <div className="relative z-10">
        <h2 className="text-3xl font-bold mb-8 text-gray-800">
          Document Management
        </h2>

        <div className="space-y-8">
          {/* Upload Box */}
          <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
            <label className="block text-gray-700 font-semibold mb-4 text-lg">
              Upload Documents
            </label>
            <div className="relative">
              <input
                type="file"
                multiple
                ref={fileInputRef}
                onChange={handleFileUpload}
                className="w-full p-4 bg-white border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white file:font-medium hover:file:bg-blue-700 transition-all duration-300"
                accept=".pdf,.doc,.docx,.txt,.md"
              />
            </div>
          </div>

          {/* Train Checkbox */}
          <div className="bg-orange-50 p-4 rounded-lg border border-orange-200">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="trainModel"
                checked={trainModel}
                onChange={(e) => setTrainModel(e.target.checked)}
                className="w-5 h-5 text-orange-600 bg-white border-2 border-orange-300 rounded focus:ring-orange-500 focus:ring-2"
              />
              <label htmlFor="trainModel" className="text-orange-800 font-medium text-sm">
                Enable model training with uploaded documents
              </label>
            </div>
          </div>

          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm animate-fadeIn">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Uploaded Files:</h3>
              <div className="space-y-3 max-h-64 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-100">
                {uploadedFiles.map((file, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-all">
                    <div className="flex items-center space-x-3">
                      <FileText size={20} className="text-blue-500" />
                      <div>
                        <span className="text-gray-800 font-medium">{file.name}</span>
                        <span className="text-gray-500 text-sm ml-2">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="text-red-500 hover:text-red-700 p-1 hover:bg-red-50 rounded transition-all"
                    >
                      <X size={16} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Loading Spinner */}
          {isUploading && (
            <div className="flex items-center justify-center p-12 bg-white rounded-lg border border-gray-200 shadow-sm">
              <div className="text-center">
                <div className="flex justify-center mb-4">
                  <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
                <p className="text-blue-700 text-lg font-semibold">Processing Upload...</p>
                <p className="text-gray-600 text-sm mt-2">Please wait while we process your documents</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const renderAnalyticsPanel = () => (
    <div className="flex-1 p-8 bg-gray-50 overflow-y-auto">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-800 mb-2">Analytics Dashboard</h2>
            <p className="text-gray-600">Vector database statistics and system insights</p>
          </div>
          <button
            onClick={loadAnalyticsData}
            disabled={analyticsLoading}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-violet-600 text-white rounded-lg hover:from-purple-700 hover:to-violet-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 shadow-md flex items-center space-x-2"
          >
            {analyticsLoading ? (
              <>
                <Loader size={18} className="animate-spin" />
                <span>Loading...</span>
              </>
            ) : (
              <>
                <Database size={18} />
                <span>Refresh Data</span>
              </>
            )}
          </button>
        </div>

        {/* Last Updated */}
        {lastUpdated && (
          <div className="mb-6 text-sm text-gray-500">
            Last updated: {lastUpdated.toLocaleString()}
          </div>
        )}

        {/* Loading State */}
        {analyticsLoading && !analyticsData && (
          <div className="flex items-center justify-center p-20">
            <div className="text-center">
              <div className="flex justify-center mb-4">
                <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
              </div>
              <p className="text-purple-700 text-xl font-semibold">Loading Analytics...</p>
              <p className="text-gray-600 mt-2">Gathering system statistics</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {analyticsError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <div className="flex items-center space-x-3">
              <X size={24} className="text-red-500" />
              <div>
                <h3 className="text-red-800 font-semibold">Error Loading Analytics</h3>
                <p className="text-red-600 mt-1">{analyticsError}</p>
              </div>
            </div>
          </div>
        )}

        {/* Analytics Content */}
        {analyticsData && !analyticsLoading && (
          <div className="space-y-8">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Total Vectors */}
              <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Total Vectors</p>
                    <p className="text-3xl font-bold text-blue-600 mt-2">
                      {analyticsData.simple?.total_vectors?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <div className="p-3 bg-blue-100 rounded-full">
                    <Database size={24} className="text-blue-600" />
                  </div>
                </div>
              </div>

              {/* Database Status */}
              <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Database Status</p>
                    <p className={`text-2xl font-bold mt-2 ${
                      analyticsData.simple?.database_status === 'healthy' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {analyticsData.simple?.database_status || 'Unknown'}
                    </p>
                  </div>
                  <div className={`p-3 rounded-full ${
                    analyticsData.simple?.database_status === 'healthy' ? 'bg-green-100' : 'bg-red-100'
                  }`}>
                    <CheckCircle size={24} className={
                      analyticsData.simple?.database_status === 'healthy' ? 'text-green-600' : 'text-red-600'
                    } />
                  </div>
                </div>
              </div>

              {/* Collections */}
              <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Collections</p>
                    <p className="text-3xl font-bold text-purple-600 mt-2">
                      {analyticsData.simple?.total_documents?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <div className="p-3 bg-purple-100 rounded-full">
                    <FileText size={24} className="text-purple-600" />
                  </div>
                </div>
              </div>

              {/* Average Similarity */}
              <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-600 text-sm font-medium">Database Size (MB)</p>
                    <p className="text-3xl font-bold text-indigo-600 mt-2">
                      {analyticsData.simple?.storage_size_mb?.toLocaleString() || '0'}
                    </p>
                  </div>
                  <div className="p-3 bg-indigo-100 rounded-full">
                    <Brain size={24} className="text-indigo-600" />
                  </div>
                </div>
              </div>
            </div>

            {/* Detailed Stats */}
            {analyticsData.collections && analyticsData.collections.length > 0 && (
              <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-xl font-bold text-gray-800 flex items-center space-x-2">
                    <Database size={20} className="text-purple-600" />
                    <span>Collections Overview</span>
                  </h3>
                </div>
                <div className="p-6">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-gray-200">
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Collection</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Vector Count</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Dimensions</th>
                          <th className="text-left py-3 px-4 font-semibold text-gray-700">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analyticsData.collections.map((collection, index) => (
                          <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-3 px-4 font-medium text-gray-800">{collection.name}</td>
                            <td className="py-3 px-4 text-gray-600">{collection.vector_count?.toLocaleString() || '0'}</td>
                            <td className="py-3 px-4 text-gray-600">{collection.dimensions || 'N/A'}</td>
                            <td className="py-3 px-4">
                              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                                collection.status === 'active' 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-yellow-100 text-yellow-800'
                              }`}>
                                {collection.status || 'Unknown'}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* Performance Metrics */}
            {analyticsData.performance && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                  <div className="p-6 border-b border-gray-200">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center space-x-2">
                      <Zap size={20} className="text-yellow-600" />
                      <span>Query Performance</span>
                    </h3>
                  </div>
                  <div className="p-6 space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Average Query Time</span>
                      <span className="font-semibold text-blue-600">
                        {analyticsData.performance.avg_query_time || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Queries</span>
                      <span className="font-semibold text-green-600">
                        {analyticsData.performance.total_queries?.toLocaleString() || '0'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Cache Hit Rate</span>
                      <span className="font-semibold text-purple-600">
                        {analyticsData.performance.cache_hit_rate 
                          ? (analyticsData.performance.cache_hit_rate * 100).toFixed(1) + '%'
                          : 'N/A'
                        }
                      </span>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                  <div className="p-6 border-b border-gray-200">
                    <h3 className="text-xl font-bold text-gray-800 flex items-center space-x-2">
                      <Sparkles size={20} className="text-pink-600" />
                      <span>System Health</span>
                    </h3>
                  </div>
                  <div className="p-6 space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Memory Usage</span>
                      <span className="font-semibold text-orange-600">
                        {analyticsData.system?.memory_usage || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Storage Used</span>
                      <span className="font-semibold text-red-600">
                        {analyticsData.system?.storage_used || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Uptime</span>
                      <span className="font-semibold text-indigo-600">
                        {analyticsData.system?.uptime || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Recent Activity */}
            {analyticsData.recent_activity && analyticsData.recent_activity.length > 0 && (
              <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                <div className="p-6 border-b border-gray-200">
                  <h3 className="text-xl font-bold text-gray-800 flex items-center space-x-2">
                    <MessageCircle size={20} className="text-blue-600" />
                    <span>Recent Activity</span>
                  </h3>
                </div>
                <div className="p-6">
                  <div className="space-y-3 max-h-64 overflow-y-auto">
                    {analyticsData.recent_activity.map((activity, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            activity.type === 'query' ? 'bg-blue-500' :
                            activity.type === 'upload' ? 'bg-green-500' :
                            'bg-gray-500'
                          }`}></div>
                          <span className="text-gray-800 font-medium">{activity.action}</span>
                        </div>
                        <span className="text-gray-500 text-sm">{activity.timestamp}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {!analyticsData && !analyticsLoading && !analyticsError && (
          <div className="text-center mt-20">
            <div className="relative mb-8">
              <Database size={80} className="mx-auto text-purple-500" />
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-4">
              No Analytics Data
            </h3>
            <p className="text-gray-600 mb-6">Click "Refresh Data" to load analytics information</p>
            <button
              onClick={loadAnalyticsData}
              className="px-6 py-3 bg-gradient-to-r from-purple-600 to-violet-600 text-white rounded-lg hover:from-purple-700 hover:to-violet-700 transition-all duration-300 shadow-md flex items-center space-x-2 mx-auto"
            >
              <Database size={18} />
              <span>Load Analytics</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-gray-100 overflow-hidden">
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out;
        }
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(243, 244, 246, 1); }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(156, 163, 175, 0.8); border-radius: 2px; }
        .scrollbar-thin::-webkit-scrollbar { width: 4px; }
        .scrollbar-track-gray-100::-webkit-scrollbar-track { background: rgb(243, 244, 246); }
        .scrollbar-thumb-gray-400::-webkit-scrollbar-thumb { background: rgb(156, 163, 175); border-radius: 2px; }
      `}</style>
      
      {renderSidePanel()}
      
      <div className="flex-1 flex flex-col">
        {activePanel === 'chat' && renderChatPanel()}
        {activePanel === 'upload' && renderUploadPanel()}
        {activePanel === 'analytics' && renderAnalyticsPanel()}
      </div>
    </div>
  );
};

export default App;