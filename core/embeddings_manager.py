# core/embeddings_manager.py - Corrected and Robust Version

import logging
import numpy as np
from typing import List, Optional
import hashlib

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Enhanced embeddings manager with fallback capabilities"""
    
    _instance = None
    _model = None
    _dimension = 384  # Default for all-MiniLM-L6-v2
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingsManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """Load embedding model with fallback options"""
        try:
            # Import here to handle potential issues
            from sentence_transformers import SentenceTransformer
            from config.settings import EMBEDDING_MODEL
            
            # Clean model name - handle both formats
            model_name = EMBEDDING_MODEL
            if model_name.startswith("sentence-transformers/"):
                model_name = model_name.replace("sentence-transformers/", "")
            
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(model_name)
            
            # Test the model and get actual dimension
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Embedding model loaded successfully with dimension {self._dimension}")
            
        except ImportError as e:
            logger.error(f"sentence-transformers not available: {e}")
            self._setup_fallback()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            logger.info("Setting up fallback embedding system...")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback hash-based embeddings"""
        logger.warning("⚠️ Using fallback hash-based embeddings")
        self._model = None
        self._dimension = 384  # Keep standard dimension
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance of EmbeddingsManager"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
def embed_texts(self, texts: List[str]) -> np.ndarray:
    """Create embeddings for multiple texts with caching support"""
    try:
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return np.array([]).reshape(0, self._dimension)

        logger.debug(f"Creating embeddings for {len(valid_texts)} texts")

        # Check cache first
        from utils.caching import rag_cache
        cached_embeddings = []
        texts_to_embed = []
        cache_hits = 0

        for text in valid_texts:
            if rag_cache.enabled:
                cached = rag_cache.get_cached_embedding(text)
                if cached is not None:
                    cached_embeddings.append(cached)
                    cache_hits += 1
                    continue
            
            # Need to generate embedding for this text
            cached_embeddings.append(None)
            texts_to_embed.append(text)

        logger.debug(f"Cache hits: {cache_hits}/{len(valid_texts)}")

        # Generate new embeddings for uncached texts
        final_embeddings = []
        new_embeddings_idx = 0
        
        if texts_to_embed:
            if self._model is not None:
                # Use sentence transformers
                new_embeddings = self._model.encode(texts_to_embed, convert_to_numpy=True)
                new_embeddings = new_embeddings.astype(np.float32)
                
                # Cache new embeddings
                if rag_cache.enabled:
                    rag_cache.batch_cache_embeddings(texts_to_embed, new_embeddings)
                
            else:
                # Use fallback method
                new_embeddings = self._fallback_embed_texts(texts_to_embed)
        
        # Combine cached and new embeddings
        for i, text in enumerate(valid_texts):
            if cached_embeddings[i] is not None:
                # Use cached embedding
                final_embeddings.append(cached_embeddings[i])
            else:
                # Use newly generated embedding
                if texts_to_embed and new_embeddings_idx < len(new_embeddings):
                    final_embeddings.append(new_embeddings[new_embeddings_idx])
                    new_embeddings_idx += 1
                else:
                    # Fallback - create zero embedding
                    final_embeddings.append(np.zeros(self._dimension, dtype=np.float32))

        result = np.array(final_embeddings, dtype=np.float32)
        logger.debug(f"Generated embeddings shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        # Return fallback embeddings on error
        return self._fallback_embed_texts(texts) if texts else np.array([]).reshape(0, self._dimension)

    
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for a single query with fallback"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            query = query.strip()
            logger.debug(f"Creating query embedding for: {query[:50]}...")
            
            if self._model is not None:
                # Use sentence transformers
                embedding = self._model.encode([query], convert_to_numpy=True)
                return embedding[0].astype(np.float32)
            else:
                # Use fallback method
                return self._fallback_embed_texts([query])[0]
                
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            # Return fallback embedding on error
            return self._fallback_embed_texts([query or ""])[0]
    
    def _fallback_embed_texts(self, texts: List[str]) -> np.ndarray:
        """Create simple hash-based embeddings as fallback"""
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding from text hash
            text_hash = hashlib.md5(text.lower().encode()).hexdigest()
            
            # Convert hash to numerical values
            embedding = []
            for i in range(0, len(text_hash), 2):
                hex_pair = text_hash[i:i+2]
                embedding.append(int(hex_pair, 16) / 255.0)  # Normalize to 0-1
            
            # Extend or truncate to required dimension
            while len(embedding) < self._dimension:
                embedding.extend(embedding[:min(self._dimension - len(embedding), len(embedding))])
            
            embedding = embedding[:self._dimension]
            embeddings.append(embedding)
        
        result = np.array(embeddings, dtype=np.float32)
        logger.debug(f"Generated fallback embeddings shape: {result.shape}")
        return result
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        return self._dimension
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are numpy arrays
            embedding1 = np.array(embedding1, dtype=np.float32)
            embedding2 = np.array(embedding2, dtype=np.float32)
            
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def batch_similarities(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarities between query and multiple documents"""
        try:
            if document_embeddings.size == 0:
                return np.array([])
            
            # Ensure proper data types
            query_embedding = np.array(query_embedding, dtype=np.float32)
            document_embeddings = np.array(document_embeddings, dtype=np.float32)
            
            # Handle single embedding case
            if len(document_embeddings.shape) == 1:
                document_embeddings = document_embeddings.reshape(1, -1)
            
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return np.zeros(len(document_embeddings))
            
            normalized_query = query_embedding / query_norm
            
            # Normalize document embeddings
            doc_norms = np.linalg.norm(document_embeddings, axis=1)
            # Avoid division by zero
            doc_norms = np.where(doc_norms == 0, 1, doc_norms)
            normalized_docs = document_embeddings / doc_norms[:, np.newaxis]
            
            # Compute cosine similarities
            similarities = np.dot(normalized_docs, normalized_query)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing batch similarities: {str(e)}")
            return np.array([])
    
    def is_model_loaded(self) -> bool:
        """Check if the actual model (not fallback) is loaded"""
        return self._model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_loaded": self._model is not None,
            "model_type": "sentence-transformers" if self._model is not None else "fallback",
            "dimension": self._dimension,
            "fallback_active": self._model is None
        }

# Create global instance
embeddings_manager = EmbeddingsManager()
