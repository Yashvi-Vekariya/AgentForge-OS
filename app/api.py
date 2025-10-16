import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import uvicorn
from app.llm_manager import LLMManager
from app.agents.agent_factory import AgentFactory
from app.direct_rag import DirectRAG
from app.multimodal.blip_helper import GeminiVisionHelper
from app.multimodal.whisper_helper import SimpleAudioHelper
from app.orchestration.advanced_crew import AdvancedCrewOrchestrator
from app.mlops.mlflow_manager import MLflowManager
from app.utils.safety import SafetyFilter
from app.utils.utils import UtilityHelper
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent AI System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
llm_manager = None
agent_factory = None
direct_rag = None
blip_helper = None
whisper_helper = None
crew_orchestrator = None
mlflow_manager = None
safety_filter = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global llm_manager, agent_factory, direct_rag
    global blip_helper, whisper_helper, crew_orchestrator, mlflow_manager, safety_filter
    
    try:
        # Initialize core components
        llm_manager = LLMManager()
        agent_factory = AgentFactory(llm_manager)
        direct_rag = DirectRAG(llm_manager)
        
        # Initialize multi-modal helpers
        blip_helper = GeminiVisionHelper()
        whisper_helper = SimpleAudioHelper()
        
        # Initialize orchestration
        crew_orchestrator = AdvancedCrewOrchestrator(llm_manager, None, agent_factory)
        
        # Initialize MLOps
        mlflow_manager = MLflowManager()
        
        # Initialize safety
        safety_filter = SafetyFilter()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Multi-Agent AI System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    components = {
        "llm_manager": llm_manager is not None,
        "agent_factory": agent_factory is not None,
        "direct_rag": direct_rag is not None,
        "blip_helper": blip_helper is not None,
        "whisper_helper": whisper_helper is not None,
        "crew_orchestrator": crew_orchestrator is not None,
        "mlflow_manager": mlflow_manager is not None,
        "safety_filter": safety_filter is not None
    }
    
    all_healthy = all(components.values())
    status = "healthy" if all_healthy else "degraded"
    
    return {
        "status": status,
        "components": components,
        "timestamp": UtilityHelper.get_timestamp()
    }

@app.post("/query_documents")
async def query_documents(files: List[UploadFile] = File(...), query: str = Form(...)):
    """Process uploaded documents and query them directly (no storage)"""
    try:
        from .utils.document_processor import DocumentProcessor
        
        if not query.strip():
            return {
                "status": "error",
                "error": "Query is required",
                "documents_processed": 0
            }
        
        processed_documents = []
        
        for file in files:
            # Validate file type
            if not DocumentProcessor.is_supported_format(file.filename):
                logger.warning(f"Skipping unsupported file: {file.filename}")
                continue
            
            try:
                # Read and extract content
                content = await file.read()
                text_content, file_type = DocumentProcessor.extract_text_from_bytes(
                    content, file.filename
                )
                
                if text_content and text_content.strip():
                    processed_documents.append({
                        "content": text_content,
                        "filename": file.filename,
                        "file_type": file_type
                    })
                else:
                    logger.warning(f"No content extracted from: {file.filename}")
                    
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                continue
        
        if not processed_documents:
            return {
                "status": "error",
                "error": "No valid documents could be processed",
                "documents_processed": 0
            }
        
        # Query documents directly
        result = direct_rag.process_and_query(processed_documents, query)
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in document query: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "documents_processed": 0
        }

@app.post("/run_workflow")
async def run_workflow(workflow_config: Dict[str, Any]):
    """Run multi-agent workflow"""
    try:
        result = await crew_orchestrator.execute_workflow_async(workflow_config)
        
        # Log workflow execution
        if mlflow_manager:
            mlflow_manager.log_agent_metrics(
                agent_name="workflow",
                task="multi_agent_workflow",
                metrics={
                    "workflow_success": 1 if result["status"] == "completed" else 0,
                    "agents_count": len(workflow_config.get('agents', []))
                }
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_agent/{agent_type}")
async def ask_agent(agent_type: str, query: Dict[str, Any]):
    """Ask specific agent a question"""
    try:
        task = query.get("question", "")
        context = query.get("context", {})
        
        # Safety check
        safety_result = safety_filter.validate_input(task)
        if not safety_result["is_valid"]:
            raise HTTPException(status_code=400, detail="Query failed safety check")
        
        agent = agent_factory.create_agent(agent_type)
        result = agent.act(task, context)
        
        # Note: Memory storage removed (using simple document store instead)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Agent type not found: {agent_type}")
    except Exception as e:
        logger.error(f"Error asking agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), task: str = "caption"):
    """Upload and process image with Gemini Vision"""
    try:
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if task == "caption":
            result = blip_helper.generate_image_caption(file_path)
        elif task == "vqa" and hasattr(file, 'question'):
            result = blip_helper.visual_question_answering(file_path, file.question)
        elif task == "describe":
            result = blip_helper.describe_image_detailed(file_path)
        else:
            result = blip_helper.generate_image_caption(file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...), task: str = "transcribe"):
    """Upload and process audio with Simple Audio Helper"""
    try:
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        if task == "transcribe":
            result = whisper_helper.transcribe_audio(file_path)
        elif task == "transcribe_with_timestamps":
            result = whisper_helper.transcribe_with_timestamps(file_path)
        elif task == "detect_language":
            result = whisper_helper.detect_language(file_path)
        else:
            result = whisper_helper.transcribe_audio(file_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_lora")
async def train_lora(training_config: Dict[str, Any], background_tasks: BackgroundTasks):
    """Start LoRA fine-tuning training"""
    try:
        from scripts.train_lora import start_training
        
        def run_training():
            start_training(training_config)
        
        background_tasks.add_task(run_training)
        
        return {
            "status": "training_started",
            "message": "LoRA training started in background",
            "config": training_config
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query(query: Dict[str, Any]):
    """Query documents directly without storage (deprecated - use /query_documents instead)"""
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use /query_documents with file uploads instead.",
        "query": query.get("question", ""),
        "response": "No documents available. Please use the /query_documents endpoint to upload and query documents directly."
    }

@app.get("/agents/available")
async def get_available_agents():
    """Get list of available agents"""
    try:
        agents = agent_factory.get_available_agents()
        return {
            "available_agents": agents,
            "total_agents": len(agents)
        }
    except Exception as e:
        logger.error(f"Error getting available agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mlflow/experiments")
async def get_mlflow_experiments(experiment_name: str = None):
    """Get MLflow experiment results"""
    try:
        results = mlflow_manager.get_experiment_results(experiment_name)
        return results
    except Exception as e:
        logger.error(f"Error getting MLflow experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/safety/check")
async def safety_check(content: Dict[str, Any]):
    """Check content safety"""
    try:
        text = content.get("text", "")
        result = safety_filter.check_content_safety(text)
        return result
    except Exception as e:
        logger.error(f"Error in safety check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)