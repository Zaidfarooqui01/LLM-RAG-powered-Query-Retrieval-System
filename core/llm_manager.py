# core/llm_manager.py - Simplified for RAG System

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class LLMManager:
    """Simplified LLM manager for RAG responses"""
    
    def __init__(self):
        self.fallback_enabled = True
        logger.info("LLM Manager initialized (fallback mode)")
    
    def generate_response_with_context(self, query: str, 
                                     retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using retrieved context"""
        try:
            if not retrieved_docs:
                return self._create_simple_response("No relevant documents found for your query.")
            
            # Extract and combine relevant text from documents
            context_text = self._extract_context_text(retrieved_docs)
            
            if not context_text:
                return self._create_simple_response("Found documents but could not extract readable content.")
            
            # Generate intelligent response using the context
            answer = self._generate_context_based_answer(query, context_text, retrieved_docs)
            
            return self._create_simple_response(answer)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._create_simple_response(f"Error generating response: {str(e)}")
    
    def _extract_context_text(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Extract and combine text from retrieved documents"""
        context_parts = []
        
        for doc in retrieved_docs[:5]:  # Use top 5 documents
            text = doc.get('text', '').strip()
            if text and len(text) > 20:
                # Clean and truncate text
                cleaned_text = re.sub(r'\s+', ' ', text)
                if len(cleaned_text) > 300:
                    cleaned_text = cleaned_text[:300] + "..."
                context_parts.append(cleaned_text)
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _generate_context_based_answer(self, query: str, context: str, 
                                     retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate answer based on context (simplified approach)"""
        try:
            # Extract key information from query
            query_keywords = self._extract_query_keywords(query)
            
            # Find most relevant sentences from context
            relevant_sentences = self._find_relevant_sentences(context, query_keywords)
            
            if not relevant_sentences:
                # Fallback: return first substantial parts of context
                sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 15]
                relevant_sentences = sentences[:3]  # First 3 sentences
            
            # Create a coherent response
            if relevant_sentences:
                answer = self._create_coherent_answer(query, relevant_sentences)
                logger.info(f"Generated context-based answer: {answer[:100]}...")
                return answer
            else:
                return "Based on the available documents, I could not find specific information to answer your query."
                
        except Exception as e:
            logger.error(f"Error in context-based answer generation: {str(e)}")
            return f"Based on the retrieved documents: {context[:200]}..."
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract key terms from the query"""
        # Remove common words and extract meaningful terms
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w{3,}\b', query.lower())  # Words with 3+ characters
        return [word for word in words if word not in stopwords]
    
    def _find_relevant_sentences(self, context: str, keywords: List[str]) -> List[str]:
        """Find sentences in context that contain query keywords"""
        if not keywords:
            return []
        
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 15]
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains any keywords
            keyword_count = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            if keyword_count > 0:
                relevant_sentences.append((sentence, keyword_count))
        
        # Sort by keyword count (most relevant first)
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the sentences (not the counts)
        return [sentence for sentence, count in relevant_sentences[:4]]  # Top 4 sentences
    
    def _create_coherent_answer(self, query: str, relevant_sentences: List[str]) -> str:
        """Create a coherent answer from relevant sentences"""
        if not relevant_sentences:
            return "No relevant information found in the documents."
        
        # If asking "What is..." or "What are...", try to provide definition-style answer
        if query.lower().startswith(('what is', 'what are')):
            # Look for definitional sentences (containing "is", "are", "means", etc.)
            definitional = [s for s in relevant_sentences if any(word in s.lower() for word in ['is ', 'are ', 'means', 'refers to', 'defined as'])]
            if definitional:
                # Use the best definitional sentence
                answer = definitional[0]
                if len(answer) < 100 and len(relevant_sentences) > 1:
                    # Add supporting information
                    answer += " " + relevant_sentences[1]
                return answer
        
        # For other queries, combine relevant sentences intelligently
        combined_answer = ". ".join(relevant_sentences[:3])  # Top 3 sentences
        
        # Ensure proper sentence ending
        if not combined_answer.endswith('.'):
            combined_answer += '.'
        
        # Limit length
        if len(combined_answer) > 600:
            combined_answer = combined_answer[:597] + "..."
        
        return combined_answer
    
    def _create_simple_response(self, answer: str) -> Dict[str, Any]:
        """Create simple response structure"""
        return {
            'justification': answer,
            'decision': 'answered',
            'confidence': 'medium',
            'sources': []
        }
    
    def generate_simple_response(self, query: str) -> str:
        """Generate simple response without context (fallback)"""
        try:
            # Simple fallback responses for common queries
            query_lower = query.lower()
            
            if 'what is' in query_lower:
                return f"I would need to search through relevant documents to provide information about that topic."
            elif 'how to' in query_lower or 'how do' in query_lower:
                return f"I would need to review the appropriate documentation to provide step-by-step guidance."
            elif 'when' in query_lower:
                return f"I would need to check the relevant documents for timing or date information."
            elif 'where' in query_lower:
                return f"I would need to search through the documents for location-specific information."
            else:
                return f"I would need to analyze relevant documents to provide a comprehensive answer to your question."
                
        except Exception as e:
            logger.error(f"Error in simple response generation: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."
    
    def extract_simple_answer(self, response: Dict[str, Any]) -> str:
        """Extract simple string answer from response"""
        if isinstance(response, dict):
            return response.get('justification', 'No answer available')
        else:
            return str(response)
    
    def is_available(self) -> bool:
        """Check if LLM manager is available"""
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM manager statistics"""
        return {
            'mode': 'simplified',
            'fallback_enabled': self.fallback_enabled,
            'status': 'active'
        }

# Create global instance
llm_manager = LLMManager()
