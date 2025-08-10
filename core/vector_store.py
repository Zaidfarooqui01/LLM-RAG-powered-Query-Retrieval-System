# core/vector_store.py - Corrected and Optimized Version

import os
import json
import pickle
import logging
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss

from core.embeddings_manager import embeddings_manager
from config.settings import VECTOR_STORE_PATH, DEFAULT_TOP_K, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class VectorStore:
    """Enhanced FAISS-based vector store with robust persistence"""
    
    def __init__(self, backend: str = "faiss"):
        self.backend = backend
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # FAISS components
        self.index = None
        self.metadata = []  # List of document data with text content
        self.doc_id_to_indices = {}  # Maps doc_id to list of indices
        
        # File paths - FIXED: Consistent naming
        self.index_path = self.vector_store_path / "faiss_index.bin"
        self.metadata_path = self.vector_store_path / "metadata.pkl"
        self.doc_mapping_path = self.vector_store_path / "doc_mapping.json"
        
        # Initialize or load
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize or load the vector store"""
        try:
            if self._can_load_from_disk():
                logger.info("Loading existing vector store from disk")
                self._load_from_disk()
            else:
                logger.info("Initializing new vector store")
                self._create_new_index()
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            logger.info("Creating fresh vector store")
            self._create_new_index()
    
    def _can_load_from_disk(self) -> bool:
        """Check if we can load existing data from disk"""
        return (self.index_path.exists() and 
                self.metadata_path.exists() and 
                self.index_path.stat().st_size > 0)
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        try:
            # Get embedding dimension
            dimension = embeddings_manager.get_embedding_dimension()
            logger.info(f"Creating new FAISS index with dimension: {dimension}")
            
            # Create FAISS index (Inner Product for cosine similarity with normalization)
            self.index = faiss.IndexFlatIP(dimension)
            self.metadata = []
            self.doc_id_to_indices = {}
            
            logger.info("✅ New vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error creating new index: {str(e)}")
            raise
    
    def _load_from_disk(self):
        """Load existing index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))
            logger.info(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"✅ Loaded {len(self.metadata)} metadata entries")
            
            # Load document mapping if available
            if self.doc_mapping_path.exists():
                with open(self.doc_mapping_path, 'r') as f:
                    self.doc_id_to_indices = json.load(f)
                logger.info(f"✅ Loaded mapping for {len(self.doc_id_to_indices)} documents")
            else:
                logger.info("Rebuilding document mapping...")
                self._rebuild_doc_mapping()
                
        except Exception as e:
            logger.error(f"Error loading from disk: {str(e)}")
            raise
    
    def _rebuild_doc_mapping(self):
        """Rebuild document ID to indices mapping"""
        self.doc_id_to_indices = {}
        for i, doc_data in enumerate(self.metadata):
            metadata = doc_data.get('metadata', {}) if isinstance(doc_data, dict) else {}
            doc_id = metadata.get('doc_id')
            if doc_id:
                if doc_id not in self.doc_id_to_indices:
                    self.doc_id_to_indices[doc_id] = []
                self.doc_id_to_indices[doc_id].append(i)
        
        logger.info(f"Rebuilt mapping for {len(self.doc_id_to_indices)} documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents with text content to the vector store"""
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            logger.info(f"📝 Adding {len(documents)} documents to vector store")
            
            # Extract texts for embedding
            texts = []
            valid_documents = []
            
            for i, doc in enumerate(documents):
                text = doc.get('text', '').strip()
                if text and len(text) > 10:  # Only process documents with substantial text
                    texts.append(text)
                    valid_documents.append(doc)
                    logger.debug(f"   Doc {i+1}: {len(text)} chars - '{text[:50]}...'")
                else:
                    logger.warning(f"   Doc {i+1}: Skipped (no substantial text)")
            
            if not valid_documents:
                logger.warning("❌ No valid documents with text content to add")
                return
            
            logger.info(f"📄 Processing {len(valid_documents)} valid documents")
            
            # Create embeddings
            embeddings = embeddings_manager.embed_texts(texts)
            
            if embeddings.size == 0:
                logger.error("❌ No embeddings created")
                return
            
            logger.info(f"🔢 Created embeddings with shape: {embeddings.shape}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            start_index = self.index.ntotal
            self.index.add(embeddings)
            logger.info(f"✅ Added {len(embeddings)} vectors to FAISS index")
            
            # Store document data with full text content
            for i, doc in enumerate(valid_documents):
                # Store complete document data
                document_data = {
                    'text': doc['text'],  # Store actual text content
                    'metadata': doc.get('metadata', {}),
                    'index': start_index + i  # Track index position
                }
                
                self.metadata.append(document_data)
                
                # Update document mapping
                doc_id = doc.get('metadata', {}).get('doc_id')
                if doc_id:
                    if doc_id not in self.doc_id_to_indices:
                        self.doc_id_to_indices[doc_id] = []
                    self.doc_id_to_indices[doc_id].append(len(self.metadata) - 1)
                
                # Debug output
                text_preview = doc['text'][:50] if doc['text'] else "NO TEXT"
                logger.debug(f"   ✅ Stored doc {i+1}: '{text_preview}...'")
            
            logger.info(f"✅ Successfully added {len(valid_documents)} documents with text content")
            logger.info(f"📊 Total vectors in store: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents with enhanced debugging"""
        try:
            logger.debug(f"🔍 SEARCH: '{query}' (top_k={top_k})")
            logger.debug(f"   Index vectors: {self.index.ntotal}, Metadata entries: {len(self.metadata)}")
            
            if self.index.ntotal == 0:
                logger.warning("❌ No vectors in index")
                return []
            
            # Create query embedding
            query_embedding = embeddings_manager.embed_query(query)
            
            # Ensure correct format for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            actual_k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, actual_k)
            
            logger.debug(f"   Search returned {len(indices[0])} results")
            logger.debug(f"   Scores: {scores[0][:3]}...")  # First 3 scores
            logger.debug(f"   Indices: {indices[0][:3]}...")  # First 3 indices
            
            # Build results with proper error handling
            results = []
            for score, idx in zip(scores[0], indices[0]):
                # Convert numpy types to Python types
                idx = int(idx)
                score = float(score)
                
                logger.debug(f"   Processing: score={score:.4f}, idx={idx}")
                
                if idx == -1:  # FAISS returns -1 for invalid results
                    logger.debug(f"   ❌ Skipping invalid index: {idx}")
                    continue
                    
                if idx >= len(self.metadata):
                    logger.warning(f"   ❌ Index {idx} out of range (max: {len(self.metadata)-1})")
                    continue
                
                try:
                    doc_data = self.metadata[idx]
                    
                    # Extract text and metadata safely
                    if isinstance(doc_data, dict):
                        text_content = doc_data.get('text', '')
                        metadata = doc_data.get('metadata', {})
                    else:
                        # Fallback for old format
                        text_content = str(doc_data)
                        metadata = {}
                    
                    # Apply filters if provided
                    if filters and not self._match_filters(metadata, filters):
                        logger.debug(f"   ⚠️ Document {idx} filtered out")
                        continue
                    
                    # Create result
                    result = {
                        'text': text_content,
                        'metadata': metadata,
                        'score': score,
                        'index': idx
                    }
                    
                    results.append(result)
                    logger.debug(f"   ✅ Added result {len(results)}: {len(text_content)} chars")
                    
                except Exception as result_error:
                    logger.warning(f"   ❌ Error processing result {idx}: {result_error}")
                    continue
            
            logger.info(f"🎯 Search completed: {len(results)} results returned")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _match_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def persist(self):
        """Save vector store to disk with enhanced error handling"""
        try:
            logger.info("💾 Persisting vector store to disk...")
            
            # Create backup of existing files if they exist
            self._backup_existing_files()
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"✅ Saved FAISS index with {self.index.ntotal} vectors")
            
            # Save metadata
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"✅ Saved {len(self.metadata)} metadata entries")
            
            # Save document mapping
            with open(self.doc_mapping_path, 'w') as f:
                json.dump(self.doc_id_to_indices, f)
            logger.info(f"✅ Saved mapping for {len(self.doc_id_to_indices)} documents")
            
            logger.info("✅ Vector store persisted successfully")
            
        except Exception as e:
            logger.error(f"❌ Error persisting vector store: {str(e)}")
            self._restore_backup_files()
            raise
    
    def _backup_existing_files(self):
        """Create backup of existing files before saving"""
        backup_suffix = ".backup"
        
        if self.index_path.exists():
            shutil.copy2(self.index_path, str(self.index_path) + backup_suffix)
        if self.metadata_path.exists():
            shutil.copy2(self.metadata_path, str(self.metadata_path) + backup_suffix)
        if self.doc_mapping_path.exists():
            shutil.copy2(self.doc_mapping_path, str(self.doc_mapping_path) + backup_suffix)
    
    def _restore_backup_files(self):
        """Restore backup files if save operation failed"""
        import shutil
        backup_suffix = ".backup"
        
        try:
            for path in [self.index_path, self.metadata_path, self.doc_mapping_path]:
                backup_path = Path(str(path) + backup_suffix)
                if backup_path.exists():
                    shutil.copy2(backup_path, path)
                    backup_path.unlink()  # Remove backup after restore
        except Exception as restore_error:
            logger.error(f"Error restoring backup files: {restore_error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics"""
        stats = {
            'total_vectors': self.index.ntotal if self.index else 0,
            'total_documents': len(self.doc_id_to_indices),
            'metadata_entries': len(self.metadata),
            'embedding_dimension': self.index.d if self.index else 0,
            'backend': self.backend,
            'store_path': str(self.vector_store_path),
            'files_exist': {
                'index': self.index_path.exists(),
                'metadata': self.metadata_path.exists(),
                'mapping': self.doc_mapping_path.exists()
            }
        }
        
        # Add file sizes if files exist
        if self.index_path.exists():
            stats['index_size_mb'] = round(self.index_path.stat().st_size / (1024*1024), 2)
        if self.metadata_path.exists():
            stats['metadata_size_mb'] = round(self.metadata_path.stat().st_size / (1024*1024), 2)
        
        return stats
    
    def clear(self):
        """Clear all data from the vector store"""
        try:
            logger.info("🗑️ Clearing vector store...")
            self._create_new_index()
            
            # Remove persisted files
            for path in [self.index_path, self.metadata_path, self.doc_mapping_path]:
                if path.exists():
                    path.unlink()
            
            logger.info("✅ Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_document_count(self) -> int:
        """Get total number of document chunks stored"""
        return len(self.metadata)
    
    def get_document_by_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document ID"""
        if doc_id not in self.doc_id_to_indices:
            return []
        
        indices = self.doc_id_to_indices[doc_id]
        return [self.metadata[idx] for idx in indices if idx < len(self.metadata)]

# Create global instance
vector_store = VectorStore()
