# utils/caching.py - Enhanced and Optimized Version

import hashlib
import json
import logging
import pickle
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class RAGCache:
    """Enhanced caching system for RAG operations with better error handling"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache directories
        self.embeddings_cache_dir = self.cache_dir / "embeddings"
        self.query_cache_dir = self.cache_dir / "queries"
        self.document_cache_dir = self.cache_dir / "documents"
        
        # Create subdirectories
        for cache_subdir in [self.embeddings_cache_dir, self.query_cache_dir, self.document_cache_dir]:
            cache_subdir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings (configurable)
        self.default_ttl = int(os.getenv("CACHE_TTL_HOURS", "1")) * 3600  # Default 1 hour
        self.max_cache_size_mb = int(os.getenv("MAX_CACHE_SIZE_MB", "500"))
        self.enabled = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        
        logger.info(f"RAG Cache initialized: enabled={self.enabled}, TTL={self.default_ttl}s, max_size={self.max_cache_size_mb}MB")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate cache key from data with better error handling"""
        try:
            if isinstance(data, str):
                content = data
            elif isinstance(data, dict):
                content = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, list):
                content = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, np.ndarray):
                content = str(data.shape) + str(data.flatten()[:10])  # Shape + first 10 elements
            else:
                content = str(data)
            
            # Create hash
            cache_key = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]  # Shorter keys
            return cache_key
            
        except Exception as e:
            logger.warning(f"Error generating cache key: {str(e)}")
            # Fallback to timestamp-based key
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """Check if cache file is still valid"""
        try:
            if not cache_file.exists():
                return False
            
            # Check file size (corrupted if 0 bytes)
            if cache_file.stat().st_size == 0:
                cache_file.unlink()  # Remove corrupted file
                return False
            
            file_age = time.time() - cache_file.stat().st_mtime
            return file_age < ttl
            
        except Exception as e:
            logger.debug(f"Cache validation error: {e}")
            return False
    
    def cache_embedding(self, text: str, embedding: np.ndarray, ttl: Optional[int] = None) -> bool:
        """Cache text embedding with size validation"""
        if not self.enabled:
            return False
            
        try:
            # Validate input
            if not text or not isinstance(embedding, np.ndarray):
                return False
            
            cache_key = self._generate_cache_key(text)
            cache_file = self.embeddings_cache_dir / f"{cache_key}.pkl"
            
            cache_data = {
                'text_hash': hashlib.md5(text.encode()).hexdigest(),  # For verification
                'embedding': embedding.astype(np.float32),  # Consistent data type
                'embedding_shape': embedding.shape,
                'timestamp': time.time(),
                'ttl': ttl or self.default_ttl
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Cached embedding for text: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching embedding: {str(e)}")
            return False
    
    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text with validation"""
        if not self.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(text)
            cache_file = self.embeddings_cache_dir / f"{cache_key}.pkl"
            
            if not self._is_cache_valid(cache_file, self.default_ttl):
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify text matches (prevent hash collisions)
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if cache_data.get('text_hash') != text_hash:
                logger.debug("Cache key collision detected, invalidating cache")
                return None
            
            # Additional TTL check
            if time.time() - cache_data['timestamp'] > cache_data.get('ttl', self.default_ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            embedding = cache_data['embedding']
            logger.debug(f"Retrieved cached embedding for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.debug(f"Could not retrieve cached embedding: {str(e)}")
            return None
    
    def cache_query_result(self, query: str, results: List[Dict[str, Any]], 
                          filters: Optional[Dict[str, Any]] = None, 
                          ttl: Optional[int] = None) -> bool:
        """Cache query results with better deduplication"""
        if not self.enabled:
            return False
            
        try:
            # Create cache key from query and filters
            cache_content = {
                'query': query.strip().lower(),  # Normalize query
                'filters': filters or {}
            }
            cache_key = self._generate_cache_key(cache_content)
            cache_file = self.query_cache_dir / f"{cache_key}.pkl"
            
            # Compact results (remove very large text content for storage efficiency)
            compact_results = []
            for result in results:
                compact_result = result.copy()
                if 'text' in compact_result and len(compact_result['text']) > 1000:
                    compact_result['text'] = compact_result['text'][:1000] + "..."
                compact_results.append(compact_result)
            
            cache_data = {
                'query': query,
                'filters': filters,
                'results': compact_results,
                'result_count': len(results),
                'timestamp': time.time(),
                'ttl': ttl or (self.default_ttl // 2)  # Shorter TTL for query results
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Cached {len(results)} query results for: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching query results: {str(e)}")
            return False
    
    def get_cached_query_result(self, query: str, 
                               filters: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results with normalization"""
        if not self.enabled:
            return None
            
        try:
            cache_content = {
                'query': query.strip().lower(),  # Normalize query
                'filters': filters or {}
            }
            cache_key = self._generate_cache_key(cache_content)
            cache_file = self.query_cache_dir / f"{cache_key}.pkl"
            
            query_ttl = self.default_ttl // 2  # Shorter TTL for queries
            if not self._is_cache_valid(cache_file, query_ttl):
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Additional TTL check
            if time.time() - cache_data['timestamp'] > cache_data.get('ttl', query_ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            logger.debug(f"Retrieved cached query results for: {query[:50]}...")
            return cache_data['results']
            
        except Exception as e:
            logger.debug(f"Could not retrieve cached query results: {str(e)}")
            return None
    
    def cache_document_processing(self, file_path: str, processed_chunks: List[Dict[str, Any]], 
                                 ttl: Optional[int] = None) -> bool:
        """Cache processed document chunks with file integrity check"""
        if not self.enabled:
            return False
            
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Use file path, size, and modification time for cache key
            file_stats = path.stat()
            cache_content = f"{file_path}_{file_stats.st_mtime}_{file_stats.st_size}"
            cache_key = self._generate_cache_key(cache_content)
            cache_file = self.document_cache_dir / f"{cache_key}.pkl"
            
            # Compact chunks for storage (keep essential data only)
            compact_chunks = []
            for chunk in processed_chunks:
                compact_chunk = {
                    'text': chunk.get('text', ''),
                    'metadata': chunk.get('metadata', {}),
                    'chunk_index': chunk.get('chunk_index', 0)
                }
                compact_chunks.append(compact_chunk)
            
            cache_data = {
                'file_path': file_path,
                'file_mtime': file_stats.st_mtime,
                'file_size': file_stats.st_size,
                'processed_chunks': compact_chunks,
                'chunk_count': len(processed_chunks),
                'timestamp': time.time(),
                'ttl': ttl or (self.default_ttl * 24)  # Longer TTL for document processing
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Cached {len(processed_chunks)} chunks for document: {path.name}")
            return True
            
        except Exception as e:
            logger.warning(f"Error caching document processing: {str(e)}")
            return False
    
    def get_cached_document_processing(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached document processing results with integrity verification"""
        if not self.enabled:
            return None
            
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            file_stats = path.stat()
            cache_content = f"{file_path}_{file_stats.st_mtime}_{file_stats.st_size}"
            cache_key = self._generate_cache_key(cache_content)
            cache_file = self.document_cache_dir / f"{cache_key}.pkl"
            
            long_ttl = self.default_ttl * 24  # 24 hours for document processing
            if not self._is_cache_valid(cache_file, long_ttl):
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify file hasn't changed
            if (cache_data.get('file_mtime') != file_stats.st_mtime or 
                cache_data.get('file_size') != file_stats.st_size):
                cache_file.unlink()  # Remove outdated cache
                logger.debug(f"Document cache invalidated due to file changes: {path.name}")
                return None
            
            # Additional TTL check
            if time.time() - cache_data['timestamp'] > cache_data.get('ttl', long_ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            logger.info(f"Retrieved cached document processing for: {path.name} ({cache_data.get('chunk_count', 0)} chunks)")
            return cache_data['processed_chunks']
            
        except Exception as e:
            logger.debug(f"Could not retrieve cached document processing: {str(e)}")
            return None
    
    def clear_cache(self, cache_type: Optional[str] = None) -> bool:
        """Clear cache files with better feedback"""
        try:
            if cache_type == 'embeddings':
                cache_dirs = [self.embeddings_cache_dir]
            elif cache_type == 'queries':
                cache_dirs = [self.query_cache_dir]
            elif cache_type == 'documents':
                cache_dirs = [self.document_cache_dir]
            else:
                cache_dirs = [self.embeddings_cache_dir, self.query_cache_dir, self.document_cache_dir]
            
            total_cleared = 0
            total_size_freed = 0
            
            for cache_subdir in cache_dirs:
                for cache_file in cache_subdir.glob("*.pkl"):
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    total_cleared += 1
                    total_size_freed += file_size
            
            size_mb = round(total_size_freed / (1024 * 1024), 2)
            logger.info(f"Cleared {total_cleared} cache files, freed {size_mb}MB")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = {
                'embeddings': self._get_cache_subdir_stats(self.embeddings_cache_dir),
                'queries': self._get_cache_subdir_stats(self.query_cache_dir),
                'documents': self._get_cache_subdir_stats(self.document_cache_dir),
                'enabled': self.enabled,
                'default_ttl_hours': self.default_ttl / 3600,
                'max_size_mb': self.max_cache_size_mb
            }
            
            # Total stats
            total_files = sum(s['files'] for s in [stats['embeddings'], stats['queries'], stats['documents']])
            total_size = sum(s['size_mb'] for s in [stats['embeddings'], stats['queries'], stats['documents']])
            
            stats['total'] = {
                'files': total_files,
                'size_mb': round(total_size, 2),
                'size_percentage': round((total_size / self.max_cache_size_mb) * 100, 1) if self.max_cache_size_mb > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'enabled': self.enabled, 'error': str(e)}
    
    def _get_cache_subdir_stats(self, cache_subdir: Path) -> Dict[str, Any]:
        """Get detailed statistics for a cache subdirectory"""
        try:
            files = list(cache_subdir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in files)
            
            # Get oldest and newest files
            if files:
                oldest_file = min(files, key=lambda f: f.stat().st_mtime)
                newest_file = max(files, key=lambda f: f.stat().st_mtime)
                oldest_age_hours = (time.time() - oldest_file.stat().st_mtime) / 3600
                newest_age_hours = (time.time() - newest_file.stat().st_mtime) / 3600
            else:
                oldest_age_hours = newest_age_hours = 0
            
            return {
                'files': len(files),
                'size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_file_age_hours': round(oldest_age_hours, 1),
                'newest_file_age_hours': round(newest_age_hours, 1)
            }
        except Exception:
            return {'files': 0, 'size_mb': 0.0, 'oldest_file_age_hours': 0, 'newest_file_age_hours': 0}
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache files with size management"""
        try:
            expired_count = 0
            size_freed = 0
            
            for cache_subdir in [self.embeddings_cache_dir, self.query_cache_dir, self.document_cache_dir]:
                for cache_file in cache_subdir.glob("*.pkl"):
                    try:
                        file_size = cache_file.stat().st_size
                        
                        with open(cache_file, 'rb') as f:
                            cache_data = pickle.load(f)
                        
                        ttl = cache_data.get('ttl', self.default_ttl)
                        if time.time() - cache_data['timestamp'] > ttl:
                            cache_file.unlink()
                            expired_count += 1
                            size_freed += file_size
                    except Exception:
                        # Remove corrupted cache files
                        try:
                            file_size = cache_file.stat().st_size
                            cache_file.unlink()
                            expired_count += 1
                            size_freed += file_size
                        except:
                            pass
            
            size_mb = round(size_freed / (1024 * 1024), 2)
            logger.info(f"Cleaned up {expired_count} expired cache files, freed {size_mb}MB")
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {str(e)}")
            return 0
    
    def batch_cache_embeddings(self, texts: List[str], embeddings: np.ndarray, 
                              ttl: Optional[int] = None) -> int:
        """Cache multiple embeddings at once with progress tracking"""
        if not self.enabled:
            return 0
            
        try:
            cached_count = 0
            
            for i, text in enumerate(texts):
                if i < len(embeddings):
                    if self.cache_embedding(text, embeddings[i], ttl):
                        cached_count += 1
            
            logger.info(f"Batch cached {cached_count}/{len(texts)} embeddings")
            return cached_count
            
        except Exception as e:
            logger.error(f"Error batch caching embeddings: {str(e)}")
            return 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cache system status"""
        stats = self.get_cache_stats()
        
        return {
            'enabled': self.enabled,
            'healthy': True,
            'cache_dir': str(self.cache_dir),
            'stats': stats,
            'warnings': self._get_warnings(stats)
        }
    
    def _get_warnings(self, stats: Dict[str, Any]) -> List[str]:
        """Generate warnings based on cache state"""
        warnings = []
        
        try:
            total_size = stats.get('total', {}).get('size_mb', 0)
            size_percentage = stats.get('total', {}).get('size_percentage', 0)
            
            if size_percentage > 90:
                warnings.append(f"Cache size critical: {size_percentage}% of max size")
            elif size_percentage > 75:
                warnings.append(f"Cache size high: {size_percentage}% of max size")
            
            # Check for very old cache files
            for cache_type in ['embeddings', 'queries', 'documents']:
                oldest_hours = stats.get(cache_type, {}).get('oldest_file_age_hours', 0)
                if oldest_hours > 72:  # Older than 3 days
                    warnings.append(f"Old {cache_type} cache files detected ({oldest_hours:.1f}h old)")
                    
        except Exception:
            pass
            
        return warnings

# Create global instance
rag_cache = RAGCache()
