# 🤖 AgentForge-OS

A powerful multi-agent AI system with a modern chat interface, powered by Google's Gemini 2.5 Flash and featuring voice interaction capabilities.

## ✨ Features

### 🎯 Multi-Agent System
- *Research Agent* - Comprehensive research and analysis
- *Development Agent* - Code generation and technical solutions
- *Data Agent* - Data analysis and insights
- *Product Agent* - Product management and strategy
- *Design Agent* - Creative design and UX guidance
- *Vision Agent* - Image analysis and visual understanding

### 🗣 Voice Interaction
- *Speech-to-Text* - Click to speak your messages
- *Text-to-Speech* - AI responses can be spoken aloud
- *Real-time Recognition* - Instant voice input processing
- *Browser-based* - No additional API keys needed for voice features

### 🎨 Modern Interface
- *Clean Design* - Modern, responsive chat interface
- *Agent Selection* - Easy switching between different AI agents
- *Real-time Chat* - Instant messaging with typing indicators
- *Voice Controls* - Toggle voice input/output easily

### 🔧 Advanced Capabilities
- *Document Upload* - RAG (Retrieval Augmented Generation)
- *Image Processing* - Upload and analyze images
- *Audio Processing* - Transcribe audio files
- *Memory System* - Persistent conversation memory
- *Safety Filters* - Built-in content safety


## 🌐 Access Points

- *Frontend*: http://localhost:3000
- *Backend API*: http://localhost:8000
- *API Documentation*: http://localhost:8000/docs
- *Health Check*: http://localhost:8000/health

## 🎮 How to Use

### Chat Interface
1. *Select Agent*: Choose from 6 different AI agents
2. *Type or Speak*: Enter your message or use voice input
3. *Voice Settings*: Toggle voice responses on/off
4. *Real-time Chat*: Get instant responses from AI agents

### Voice Features
- *🎤 Voice Input*: Click microphone button to speak
- *🔊 Voice Output*: Toggle to hear AI responses
- *🌐 Browser-based*: Uses Web Speech API (no extra setup)

### Agent Modes
- *Research*: Deep analysis and research tasks
- *Dev*: Code generation and technical help
- *Data*: Data analysis and insights
- *Product*: Product strategy and management
- *Design*: Creative and UX guidance
- *Vision*: Image analysis and visual tasks


## 📡 API Endpoints

### Core Endpoints
- GET /health - System health check
- POST /ask_agent/{agent_type} - Chat with specific agent
- GET /agents/available - List available agents

### File Upload
- POST /upload_image - Process images
- POST /upload_audio - Transcribe audio
- POST /upload_doc - Upload documents for RAG

### Advanced Features
- POST /rag/query - Query uploaded documents
- GET /memory/query - Search conversation memory
- POST /safety/check - Content safety check

## 🛠 Development

### Backend Structure

app/
├── api.py              # FastAPI main application
├── llm_manager.py      # Gemini LLM integration
├── agents/             # AI agent implementations
├── multimodal/         # Image/audio processing
├── rag/               # Document retrieval system
└── utils/             # Utilities and helpers


### Frontend Structure

frontend/src/
├── components/         # React components
├── services/          # API service layer
└── index.css         # Modern styling


### Adding New Agents
1. Create agent class in app/agents/
2. Register in agent_factory.py
3. Add to frontend agent selection

## 🔒 Security & Safety

- *Content Filtering*: Built-in safety filters
- *API Key Security*: Environment variable storage
- *CORS Protection*: Configurable CORS settings
- *Input Validation*: Request validation and sanitization

## 🌍 Browser Compatibility

### Voice Features Support
- ✅ Chrome 25+
- ✅ Edge 79+
- ✅ Firefox 62+
- ✅ Safari 14.1+

### Requirements for Voice
- HTTPS connection (or localhost)
- Microphone permissions
- Modern browser with Web Speech API


## 📚 Documentation

- *[API Setup Guide](API_SETUP_GUIDE.md)* - Detailed API configuration
- *[API Documentation](http://localhost:8000/docs)* - Interactive API docs
- *[Configuration Guide](config/config.yaml)* - System configuration

## 🎯 Use Cases

### Research & Analysis
- Market research and competitive analysis
- Academic research assistance
- Data analysis and insights

### Development
- Code generation and review
- Technical documentation
- Architecture planning

### Content Creation
- Writing assistance
- Design guidance
- Product strategy

### Multimodal Tasks
- Image analysis and description
- Audio transcription
- Document processing

## 🚀 Performance

- *Response Time*: < 2 seconds for most queries
- *Concurrent Users*: Supports multiple simultaneous users
- *Memory Efficient*: Optimized for resource usage
- *Scalable*: Easily deployable to cloud platforms

## 👤 Author

Yashvi Vekariya  
🌐 [LinkedIn](https://www.linkedin.com/in/yashvi-vekariya)  
💻 [GitHub](https://github.com/Yashvi-Vekariya)  
📧 [yashviivekariya@gmail.com](mailto:yashviivekariya@gmail.com)
