# core/explainability.py - Simplified and Robust Version

import logging
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ExplainabilityManager:
    """Simplified explainability manager for RAG responses"""
    
    def __init__(self):
        self.max_quote_length = 200
        self.min_sentence_length = 15
    
    def enhance_response_with_sources(self, response: str, retrieved_docs: List[Dict[str, Any]], 
                                    query: str) -> Dict[str, Any]:
        """Enhance simple response with source information"""
        try:
            if not retrieved_docs:
                return {
                    'response': response,
                    'sources': [],
                    'confidence': 'low',
                    'explanation': 'No source documents available'
                }
            
            # Extract source information
            sources = self._extract_source_info(retrieved_docs, response)
            
            # Assess confidence
            confidence = self._assess_confidence(sources, response)
            
            # Create explanation
            explanation = self._create_simple_explanation(sources, response, query)
            
            return {
                'response': response,
                'sources': sources,
                'confidence': confidence,
                'explanation': explanation,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error enhancing response with sources: {str(e)}")
            return {
                'response': response,
                'sources': [],
                'confidence': 'low',
                'explanation': f'Error processing sources: {str(e)}'
            }
    
    def _extract_source_info(self, retrieved_docs: List[Dict[str, Any]], 
                           response: str) -> List[Dict[str, Any]]:
        """Extract and format source information"""
        sources = []
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Top 3 sources
            try:
                metadata = doc.get('metadata', {})
                doc_text = doc.get('text', '')
                score = doc.get('score', 0.0)
                
                # Find best quote from document
                quote = self._find_relevant_quote(doc_text, response)
                
                source_info = {
                    'rank': i + 1,
                    'source_file': self._format_filename(metadata.get('source', '')),
                    'chunk_index': metadata.get('chunk_index', i),
                    'doc_id': metadata.get('doc_id', f'doc_{i}'),
                    'similarity_score': round(float(score), 3),
                    'quote': quote,
                    'relevance': self._calculate_simple_relevance(doc_text, response)
                }
                
                sources.append(source_info)
                
            except Exception as e:
                logger.warning(f"Error processing document {i}: {str(e)}")
                continue
        
        return sources
    
    def _find_relevant_quote(self, doc_text: str, response: str) -> str:
        """Find most relevant quote from document text"""
        if not doc_text:
            return "No text content available"
        
        # Split into sentences
        sentences = self._split_into_sentences(doc_text)
        
        if not sentences:
            # Return first part of text if no sentences found
            return self._truncate_text(doc_text, self.max_quote_length)
        
        # Find best matching sentence using simple keyword overlap
        best_sentence = ""
        best_score = 0
        
        response_words = set(word.lower() for word in response.split() if len(word) > 3)
        
        for sentence in sentences:
            if len(sentence) < self.min_sentence_length:
                continue
                
            sentence_words = set(word.lower() for word in sentence.split())
            
            # Calculate word overlap score
            common_words = response_words.intersection(sentence_words)
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # If no good match, use first substantial sentence
        if not best_sentence:
            for sentence in sentences:
                if len(sentence) >= self.min_sentence_length:
                    best_sentence = sentence
                    break
        
        # Final fallback
        if not best_sentence:
            best_sentence = doc_text[:self.max_quote_length]
        
        return self._truncate_text(best_sentence, self.max_quote_length)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
    
    def _calculate_simple_relevance(self, doc_text: str, response: str) -> str:
        """Calculate simple relevance score"""
        if not doc_text or not response:
            return "low"
        
        # Count common words (simple approach)
        doc_words = set(word.lower() for word in doc_text.split() if len(word) > 3)
        response_words = set(word.lower() for word in response.split() if len(word) > 3)
        
        if not doc_words or not response_words:
            return "low"
        
        common_words = doc_words.intersection(response_words)
        overlap_ratio = len(common_words) / min(len(doc_words), len(response_words))
        
        if overlap_ratio > 0.3:
            return "high"
        elif overlap_ratio > 0.15:
            return "medium"
        else:
            return "low"
    
    def _format_filename(self, filepath: str) -> str:
        """Extract and format filename from path"""
        if not filepath:
            return "Unknown source"
        
        filename = os.path.basename(filepath)
        # Remove file extension and cleanup
        name_without_ext = os.path.splitext(filename)[0]
        
        # Clean up auto-generated names
        if name_without_ext.startswith("doc_"):
            return f"Document {name_without_ext.split('_')[-1][:8]}"
        
        return name_without_ext if name_without_ext else filename
    
    def _assess_confidence(self, sources: List[Dict[str, Any]], response: str) -> str:
        """Assess confidence in the response"""
        if not sources:
            return "low"
        
        high_relevance_count = sum(1 for s in sources if s.get('relevance') == 'high')
        avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
        
        # Simple confidence assessment
        if high_relevance_count >= 2 or avg_similarity > 0.7:
            return "high"
        elif high_relevance_count >= 1 or avg_similarity > 0.4:
            return "medium"
        else:
            return "low"
    
    def _create_simple_explanation(self, sources: List[Dict[str, Any]], 
                                 response: str, query: str) -> str:
        """Create simple explanation of the response"""
        if not sources:
            return "No source documents were found to support this response."
        
        explanation_parts = [
            f"This response was generated based on {len(sources)} retrieved document(s)."
        ]
        
        high_relevance = [s for s in sources if s.get('relevance') == 'high']
        if high_relevance:
            explanation_parts.append(
                f"{len(high_relevance)} source(s) showed high relevance to your query."
            )
        
        # Mention top source
        top_source = sources[0] if sources else None
        if top_source:
            source_name = top_source.get('source_file', 'Unknown')
            similarity = top_source.get('similarity_score', 0)
            explanation_parts.append(
                f"The primary source '{source_name}' had a similarity score of {similarity:.2f}."
            )
        
        return " ".join(explanation_parts)
    
    def create_audit_log(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create simple audit log"""
        return {
            'timestamp': datetime.now().isoformat(),
            'query': query[:100] + "..." if len(query) > 100 else query,
            'response_length': len(response),
            'sources_count': len(sources),
            'confidence_distribution': {
                'high': sum(1 for s in sources if s.get('relevance') == 'high'),
                'medium': sum(1 for s in sources if s.get('relevance') == 'medium'),
                'low': sum(1 for s in sources if s.get('relevance') == 'low')
            },
            'avg_similarity': round(
                sum(s.get('similarity_score', 0) for s in sources) / len(sources), 3
            ) if sources else 0.0
        }
    
    def format_sources_for_display(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for human-readable display"""
        if not sources:
            return "No sources available"
        
        formatted_sources = []
        for source in sources:
            source_line = f"• {source.get('source_file', 'Unknown')} " \
                         f"(Relevance: {source.get('relevance', 'unknown')}, " \
                         f"Similarity: {source.get('similarity_score', 0):.2f})"
            formatted_sources.append(source_line)
        
        return "\n".join(formatted_sources)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get explainability manager statistics"""
        return {
            'max_quote_length': self.max_quote_length,
            'min_sentence_length': self.min_sentence_length,
            'status': 'active'
        }

# Create global instance
explainability_manager = ExplainabilityManager()
