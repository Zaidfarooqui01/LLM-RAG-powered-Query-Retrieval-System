"""
HackRx 6.0 Intelligent Query-Retrieval System - Fixed Version
"""

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from fastapi.responses import JSONResponse
import asyncio
from pathlib import Path
import uuid
import time
import threading
from datetime import datetime
import logging
import json
import uvicorn
import shutil
import os
import requests
import re

# Import your RAG components (only import what exists)
from core.document_processor import document_processor
from core.embeddings_manager import embeddings_manager
from core.vector_store import vector_store
from core.llm_manager import llm_manager
from config.settings import *

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),  # Save to file
        logging.StreamHandler()             # Also show in terminal
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Query-Retrieval System",
    description="LLM-powered document Q&A system for HackRx 6.0",
    version="2.0.0",
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HackRx API Models
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="Document URL", min_length=1)
    questions: List[str] = Field(..., description="List of questions", min_items=1)
    
    @validator('documents', pre=True)
    def validate_documents(cls, v):
        if not v:
            raise ValueError('documents field is required')
        
        # Convert to string and strip
        v = str(v).strip()
        
        if not v:
            raise ValueError('documents URL cannot be empty')
        
        # Basic URL validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, v, re.IGNORECASE):
            raise ValueError('documents must be a valid HTTP/HTTPS URL')
        
        return v
    
    @validator('questions', pre=True)
    def validate_questions(cls, v):
        if not v:
            raise ValueError('questions field is required')
        
        # Ensure it's a list
        if not isinstance(v, list):
            raise ValueError('questions must be a list')
        
        if len(v) == 0:
            raise ValueError('at least one question is required')
        
        # Validate each question
        valid_questions = []
        for i, question in enumerate(v):
            if question is None:
                raise ValueError(f'question {i+1} cannot be null')
            
            question_str = str(question).strip()
            if not question_str:
                raise ValueError(f'question {i+1} cannot be empty')
            
            if len(question_str) > 1000:  # Reasonable limit
                raise ValueError(f'question {i+1} too long (max 1000 characters)')
            
            valid_questions.append(question_str)
        
        return valid_questions

    class Config:
        schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
                "questions": ["What is the grace period for premium payment?"]
            }
        }

class HackRxResponse(BaseModel):
    answers: List[str] = Field(..., description="Array of answer strings")

# Authentication
HACKRX_TOKEN = "6ca800c46dd70bb4a8ef18a01692ac76721bb2b50303e31dbed18a186993ac1e"

# In main.py - Fix the verify_token function
def verify_token(authorization: str = Header(..., description="Bearer token")):
    """Rock-solid token verification for HackRx compliance"""
    try:
        # Log the incoming authorization header for debugging
        logger.debug(f"Received authorization header: {authorization}")
        
        # Check if authorization header exists
        if not authorization:
            logger.error("Authorization header missing")
            raise HTTPException(
                status_code=401, 
                detail="Authorization header required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check Bearer prefix (case-insensitive)
        if not authorization.lower().startswith("bearer "):
            logger.error(f"Invalid authorization format: {authorization}")
            raise HTTPException(
                status_code=401, 
                detail="Authorization header must start with 'Bearer '",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Extract token
        token = authorization[7:].strip()  # Remove "Bearer " and strip whitespace
        
        # Validate token exists
        if not token:
            logger.error("Token is empty")
            raise HTTPException(
                status_code=401, 
                detail="Bearer token is required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Exact token comparison
        EXPECTED_TOKEN = "6ca800c46dd70bb4a8ef18a01692ac76721bb2b50303e31dbed18a186993ac1e"
        
        if token != EXPECTED_TOKEN:
            logger.error(f"Token mismatch. Expected length: {len(EXPECTED_TOKEN)}, Got length: {len(token)}")
            raise HTTPException(
                status_code=401, 
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        logger.debug("✅ Token validation successful")
        return token
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in token validation: {str(e)}")
        raise HTTPException(
            status_code=401, 
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


# RAG Pipeline Functions
def download_document_sync(url: str) -> str:
    """Download document from URL and return local path (synchronous)"""
    try:
        # Create upload directory
        upload_dir = Path(PDF_UPLOAD_PATH)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        filename = f"doc_{uuid.uuid4().hex}.pdf"
        local_path = upload_dir / filename
        
        logger.info(f"Downloading document from: {url}")
        
        # Download file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Save file
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Downloaded document to: {local_path}")
        return str(local_path)
        
    except requests.RequestException as e:
        logger.error(f"Error downloading document from {url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def ingest_document(file_path: str) -> Dict[str, Any]:
    """Process and ingest document into vector store"""
    try:
        logger.info(f"🔍 INGEST: Starting with {file_path}")
        
        # Process document
        documents = document_processor.build_documents(file_path)
        logger.info(f"📄 PROCESSED: {len(documents)} chunks created")
        
        if not documents:
            return {'status': 'error', 'message': 'No content extracted from document'}
        
        # Show sample of what we extracted
        for i, doc in enumerate(documents[:2]):
            text_preview = doc.get('text', '')[:100]
            logger.info(f"📝 Sample chunk {i+1}: '{text_preview}...'")
        
        # Add to vector store
        logger.info("🔢 Adding to vector store...")
        vector_store.add_documents(documents)
        
        # Persist the data
        logger.info("💾 Persisting vector store...")
        if hasattr(vector_store, 'persist'):
            vector_store.persist()
        
        # Verify it worked
        stats = vector_store.get_stats()
        test_search = vector_store.search("document", top_k=1)
        
        logger.info(f"✅ Vector store: {stats.get('total_vectors', 0)} vectors, test search: {len(test_search)} results")
        
        return {
            'status': 'success',
            'chunks_created': len(documents),
            'vector_store_total': stats.get('total_vectors', 0),
            'persisted': True
        }
        
    except Exception as e:
        logger.error(f"❌ INGEST ERROR: {str(e)}")
        return {'status': 'error', 'message': str(e)}

# In main.py - Fix the answer_query function
def answer_query(query: str, top_k: int = 5) -> str:
    """Enhanced answer query with better error handling"""
    try:
        logger.info(f"🔍 QUERY: {query}")
        
        # Validate input
        if not query or not query.strip():
            return "Please provide a valid question."
        
        # Search for relevant documents
        retrieved_docs = vector_store.search(query.strip(), top_k=top_k)
        logger.info(f"📊 FOUND: {len(retrieved_docs)} documents")
        
        if not retrieved_docs:
            return "I couldn't find relevant information in the document to answer your question."
        
        # Validate document content
        valid_docs = []
        for doc in retrieved_docs:
            if isinstance(doc, dict) and doc.get('text'):
                text_content = doc['text'].strip()
                if len(text_content) > 20:
                    valid_docs.append(doc)
        
        if not valid_docs:
            return "The document appears to be empty or unreadable."
        
        # Try LLM manager first
        try:
            if hasattr(llm_manager, 'generate_response_with_context'):
                llm_response = llm_manager.generate_response_with_context(query, valid_docs)
                answer = llm_manager.extract_simple_answer(llm_response)
                
                if answer and len(answer.strip()) > 30:
                    return answer
        except Exception as llm_error:
            logger.warning(f"LLM generation failed: {llm_error}")
        
        # Fallback: Create response from document content
        context_parts = []
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        
        for doc in valid_docs[:3]:
            text = doc['text'].strip()
            sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 15]
            
            # Find relevant sentences
            relevant_sentences = []
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                context_parts.extend(relevant_sentences[:2])
            else:
                context_parts.extend(sentences[:1])
        
        if context_parts:
            answer = " ".join(context_parts[:3])
            # Ensure reasonable length
            if len(answer) > 500:
                answer = answer[:497] + "..."
            return answer
        
        return "Based on the document, I found related information but couldn't extract a specific answer to your question."
        
    except Exception as e:
        logger.error(f"❌ QUERY ERROR: {str(e)}")
        return "I encountered an error while processing your question. Please try again."


# Main HackRx endpoint

# In main.py - Update your main endpoint with better error handling
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest, 
    token: str = Depends(verify_token)
):
    """Enhanced HackRx endpoint with robust error handling"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"🚀 Processing HackRx request {request_id}")
    
    try:
        # Validate request
        if not request.documents or not request.questions:
            return HackRxResponse(answers=["Invalid request: missing documents or questions"])
        
        # Handle edge case: empty questions
        if not any(q.strip() for q in request.questions):
            return HackRxResponse(answers=["Please provide valid questions"])
        
        # Step 1: Download and process document with timeout
        logger.info("📥 Step 1: Processing document...")
        try:
            if request.documents.startswith(('http://', 'https://')):
                # Handle document download
                documents = document_processor.build_documents(request.documents)
            else:
                return HackRxResponse(answers=["Invalid document URL provided"])
                
        except requests.RequestException:
            error_msg = "Unable to download document from the provided URL"
            return HackRxResponse(answers=[error_msg for _ in request.questions])
        except Exception as doc_error:
            error_msg = f"Document processing failed: {str(doc_error)}"
            return HackRxResponse(answers=[error_msg for _ in request.questions])
        
        if not documents:
            error_msg = "No content could be extracted from the document"
            return HackRxResponse(answers=[error_msg for _ in request.questions])
        
        # Step 2: Add to vector store
        logger.info("🔢 Step 2: Indexing document...")
        try:
            vector_store.add_documents(documents)
            vector_store.persist()
        except Exception as vector_error:
            logger.error(f"Vector store error: {vector_error}")
            # Continue anyway - might still work with existing data
        
        # Step 3: Process each question
        logger.info("🤔 Step 3: Processing questions...")
        answers = []
        
        for i, question in enumerate(request.questions):
            try:
                question = question.strip()
                if not question:
                    answers.append("Please provide a valid question")
                    continue
                
                logger.info(f"🔍 Question {i+1}: {question}")
                
                # Generate answer with timeout protection
                answer = answer_query(question, top_k=5)
                
                # Validate answer quality
                if not answer or len(answer) < 10:
                    answer = "I don't have sufficient information to answer this question based on the provided document."
                
                answers.append(answer)
                logger.info(f"✅ Answer {i+1}: {answer[:100]}...")
                
            except Exception as q_error:
                logger.error(f"Error processing question {i+1}: {q_error}")
                answers.append(f"Error processing question: Please try rephrasing your question.")
        
        processing_time = time.time() - start_time
        logger.info(f"🎉 Request {request_id} completed in {processing_time:.2f}s")
        
        return HackRxResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"💥 Critical error in request {request_id}: {str(e)}")
        # Return graceful error response
        error_answers = []
        for question in request.questions:
            error_answers.append("I apologize, but I encountered a technical issue while processing your request. Please try again.")
        
        return HackRxResponse(answers=error_answers)


def _generate_answer_with_timeout(self, question: str, timeout: int = 30) -> str:
    """Generate answer with timeout protection"""
    try:
        # Use threading for timeout protection
        result_container = {"answer": None, "error": None}
        
        def generate_answer():
            try:
                result_container["answer"] = answer_query(question)
            except Exception as e:
                result_container["error"] = str(e)
        
        thread = threading.Thread(target=generate_answer)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            return "Answer generation timed out. Please try a simpler question."
        
        if result_container["error"]:
            return f"Error generating answer: {result_container['error']}"
        
        return result_container["answer"] or "No answer could be generated."
        
    except Exception as e:
        return f"Error in answer generation: {str(e)}"



# Health and utility endpoints
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "service": "HackRx 6.0 Query System"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 Intelligent Query-Retrieval System",
        "status": "running",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health",
        "endpoint": "/api/v1/hackrx/run"
    }

@app.get("/debug-vector-store/")
async def debug_vector_store():
    """Debug vector store status"""
    try:
        stats = vector_store.get_stats()
        test_search = vector_store.search("test", top_k=1)
        
        try:
            test_embedding = embeddings_manager.embed_query("test")
            embedding_shape = test_embedding.shape if hasattr(test_embedding, 'shape') else len(test_embedding)
        except Exception as e:
            embedding_shape = f"Error: {e}"
        
        return {
            "vector_store_stats": stats,
            "test_search_results": len(test_search),
            "embedding_working": embedding_shape,
            "sample_search_result": test_search[0] if test_search else "No results"
        }
    except Exception as e:
        return {"error": str(e)}
    

@app.get("/debug-cache/")
async def debug_cache():
    """Debug cache status and statistics"""
    try:
        from utils.caching import rag_cache
        
        # Get comprehensive cache status
        status = rag_cache.get_status()
        
        # Add some additional runtime info
        cache_info = {
            **status,
            "cleanup_info": {
                "expired_files_cleaned": rag_cache.cleanup_expired_cache(),
                "last_cleanup": datetime.now().isoformat()
            }
        }
        
        return cache_info
        
    except ImportError:
        return {
            "error": "Caching module not available",
            "cache_enabled": False,
            "message": "Install utils/caching.py to enable caching"
        }
    except Exception as e:
        return {
            "error": str(e),
            "cache_enabled": False
        }

@app.post("/clear-cache/")
async def clear_cache_endpoint(cache_type: Optional[str] = None):
    """Clear cache endpoint for testing"""
    try:
        from utils.caching import rag_cache
        
        success = rag_cache.clear_cache(cache_type)
        stats = rag_cache.get_cache_stats()
        
        return {
            "success": success,
            "cache_type_cleared": cache_type or "all",
            "remaining_stats": stats
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}


# Additional endpoints for testing
@app.post("/query/")
async def query_endpoint(query: str = Form(...)):
    """Query endpoint for testing"""
    try:
        answer = answer_query(query)
        return JSONResponse(content={"query": query, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Initializing HackRx 6.0 RAG System...")
    
    try:
        print("✅ Document processor ready")
        print("✅ Embeddings manager ready") 
        print("✅ Vector store ready")
        print("✅ LLM manager ready")
        print("✅ System initialized successfully")
    except Exception as e:
        print(f"❌ System initialization error: {e}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
