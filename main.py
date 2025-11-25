# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import uuid
import logging

from src.document_processor import DocumentProcessor
from src.embedding_service import EmbeddingService
from src.vector_db import VectorDatabase
from src.memory_system import MemorySystem
from src.qa_engine import QAEngine
from src.learning_pipeline import LearningPipeline
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Intelligent Document Q&A System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_db = VectorDatabase()
memory_system = MemorySystem(vector_db)
qa_engine = QAEngine(embedding_service, vector_db, memory_system)
learning_pipeline = LearningPipeline(vector_db, qa_engine)

# Data models
class QueryRequest(BaseModel):
    query: str
    session_id: str
    document_filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict]
    processing_time: float
    session_id: str

class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback_type: str  # 'rating', 'correction', 'thumbs_up', 'thumbs_down'
    feedback_data: Dict[str, Any]
    corrected_answer: Optional[str] = None

class UploadResponse(BaseModel):
    document_id: str
    chunks_processed: int
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Intelligent Document Q&A System starting up...")
    
    # Create necessary directories
    os.makedirs("./data/documents", exist_ok=True)
    os.makedirs("./data/feedback", exist_ok=True)
    os.makedirs("./data/models", exist_ok=True)

@app.post("/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Generate document ID
        doc_id = str(uuid.uuid4())
        file_path = f"./data/documents/{doc_id}_{file.filename}"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        chunks = document_processor.process_document(file_path, doc_id)
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts)
        
        # Store in vector database
        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                'content': chunk.content,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id
            })
        
        vector_db.store_document_chunks(chunk_dicts, embeddings)
        
        return UploadResponse(
            document_id=doc_id,
            chunks_processed=len(chunks),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query"""
    try:
        result = qa_engine.process_query(
            request.query, 
            request.session_id, 
            request.document_filters
        )
        
        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            sources=result['sources'],
            processing_time=result['processing_time'],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback"""
    try:
        qa_engine.provide_feedback(
            request.interaction_id,
            request.feedback_type,
            request.feedback_data,
            request.corrected_answer
        )
        
        return {"status": "feedback_received"}
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation-history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        history = memory_system.get_episodic_memory(session_id)
        return {"session_id": session_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn-from-feedback")
async def trigger_learning(background_tasks: BackgroundTasks):
    """Trigger learning from accumulated feedback"""
    try:
        background_tasks.add_task(learning_pipeline.process_feedback_batch)
        return {"status": "learning_triggered"}
        
    except Exception as e:
        logger.error(f"Error triggering learning: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "document_qa_system"}

@app.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        # Basic metrics - extend with more sophisticated monitoring
        doc_collection = vector_db.client.get_collection("document_chunks")
        interaction_collection = vector_db.client.get_collection("user_interactions")
        
        return {
            "documents_processed": doc_collection.count(),
            "total_interactions": interaction_collection.count(),
            "active_sessions": len(memory_system.sessions),
            "cache_size": len(qa_engine.qa_cache)
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)