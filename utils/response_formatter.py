# utils/response_formatter.py - Enhanced for HackRx RAG System

import json
import logging
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """Enhanced response formatter optimized for HackRx RAG system"""
    
    def __init__(self):
        self.standard_fields = [
            'decision', 'amount', 'justification', 'sources', 'confidence'
        ]
        self.max_response_length = 600  # HackRx optimal response length
    
    def format_simple_answer(self, answer: str, sources: Optional[List[Dict[str, Any]]] = None, 
                           query: str = "", confidence: str = "medium") -> str:
        """Format simple string answer for HackRx (primary method)"""
        try:
            if not answer or not answer.strip():
                return "I don't have sufficient information to answer this question."
            
            # Clean and optimize the answer
            cleaned_answer = self._clean_and_optimize_answer(answer)
            
            # Add source attribution if available and answer is substantial
            if sources and len(cleaned_answer) > 50:
                attribution = self._create_simple_attribution(sources)
                if attribution:
                    # Only add attribution if it doesn't make response too long
                    combined = f"{cleaned_answer} {attribution}"
                    if len(combined) <= self.max_response_length:
                        cleaned_answer = combined
            
            return cleaned_answer
            
        except Exception as e:
            logger.error(f"Error formatting simple answer: {str(e)}")
            return f"Based on the available documents: {str(answer)[:400]}..."
    
    def _clean_and_optimize_answer(self, answer: str) -> str:
        """Clean and optimize answer for HackRx submission"""
        if not answer:
            return "No information available."
        
        # Remove excessive whitespace and normalize
        cleaned = ' '.join(answer.split())
        
        # Remove redundant phrases that add no value
        redundant_phrases = [
            "based on the provided documents",
            "according to the text",
            "the document states that",
            "as mentioned in the document",
            "from the information provided"
        ]
        
        for phrase in redundant_phrases:
            cleaned = re.sub(phrase, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s*,\s*,', ',', cleaned)
        cleaned = re.sub(r'\s*\.\s*\.', '.', cleaned)
        cleaned = cleaned.strip()
        
        # Ensure proper sentence structure
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
        
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        # Truncate if too long
        if len(cleaned) > self.max_response_length:
            # Try to truncate at sentence boundary
            sentences = cleaned.split('.')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '.') <= self.max_response_length - 3:
                    truncated += sentence + '.'
                else:
                    break
            
            if truncated:
                cleaned = truncated
            else:
                # Hard truncate if no good sentence boundary
                cleaned = cleaned[:self.max_response_length - 3] + "..."
        
        return cleaned
    
    def _create_simple_attribution(self, sources: List[Dict[str, Any]]) -> str:
        """Create simple source attribution"""
        if not sources:
            return ""
        
        try:
            # Get top source with highest relevance/similarity
            top_source = max(sources, key=lambda x: x.get('similarity_score', 0))
            source_name = self._format_simple_source_name(top_source.get('source', ''))
            
            if source_name:
                if len(sources) == 1:
                    return f"(Source: {source_name})"
                else:
                    return f"(Sources: {source_name} and {len(sources)-1} others)"
            
        except Exception as e:
            logger.debug(f"Error creating attribution: {e}")
        
        return ""
    
    def _format_simple_source_name(self, source_path: str) -> str:
        """Format source path to simple readable name"""
        if not source_path:
            return ""
        
        try:
            filename = os.path.basename(source_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Clean up auto-generated names
            if name_without_ext.startswith("doc_"):
                return "Document"
            
            # Clean and shorten name
            clean_name = name_without_ext.replace('_', ' ').replace('-', ' ')
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()
            
            # Limit length
            if len(clean_name) > 30:
                clean_name = clean_name[:27] + "..."
            
            return clean_name.title() if clean_name else "Document"
            
        except Exception:
            return "Document"
    
    def format_hackrx_response(self, decision: str, amount: Optional[str], 
                              justification: str, sources: List[Dict[str, Any]], 
                              confidence: str = 'medium', 
                              additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format structured response for advanced use cases"""
        
        response = {
            'decision': self._validate_decision(decision),
            'amount': self._format_amount(amount),
            'justification': self.format_simple_answer(justification, sources),
            'sources': self._format_sources(sources),
            'confidence': self._validate_confidence(confidence),
            'timestamp': datetime.now().isoformat(),
            'response_id': self._generate_response_id()
        }
        
        # Add additional information if provided
        if additional_info:
            response.update(additional_info)
        
        # Add metadata
        response['metadata'] = self._create_metadata(sources)
        
        return response
    
    def _validate_decision(self, decision: str) -> str:
        """Validate and normalize decision values"""
        valid_decisions = ['approved', 'rejected', 'insufficient_evidence', 'answered', 'pending']
        
        decision = decision.lower().strip() if decision else ''
        
        # Map common variations
        decision_mapping = {
            'approve': 'approved',
            'accept': 'approved', 
            'yes': 'approved',
            'reject': 'rejected',
            'deny': 'rejected',
            'no': 'rejected',
            'insufficient': 'insufficient_evidence',
            'unclear': 'insufficient_evidence',
            'unknown': 'insufficient_evidence',
            'answer': 'answered',
            'response': 'answered'
        }
        
        decision = decision_mapping.get(decision, decision)
        
        return decision if decision in valid_decisions else 'answered'
    
    def _format_amount(self, amount: Optional[str]) -> Optional[str]:
        """Format and validate amount field"""
        if not amount:
            return None
        
        # Clean amount string
        amount_str = str(amount).strip()
        
        # Extract number patterns
        number_pattern = r'[\d,]+(?:\.\d{2})?'
        currency_symbols = ['$', '₹', '€', '£', 'USD', 'INR', 'EUR', 'GBP']
        
        # Look for amount patterns
        for symbol in currency_symbols:
            pattern = f'{re.escape(symbol)}\\s*({number_pattern})'
            match = re.search(pattern, amount_str, re.IGNORECASE)
            if match:
                return f"{symbol}{match.group(1)}"
        
        # Look for plain numbers
        match = re.search(number_pattern, amount_str)
        if match:
            return match.group(0)
        
        # Return original if no pattern found (but truncate)
        return str(amount)[:50]
    
    def _clean_justification(self, justification: str) -> str:
        """Clean and format justification text (legacy method)"""
        return self.format_simple_answer(justification)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format source information for structured responses"""
        if not sources:
            return []
        
        formatted_sources = []
        
        for i, source in enumerate(sources[:5]):  # Limit to top 5
            try:
                formatted_source = {
                    'rank': i + 1,
                    'doc_id': source.get('doc_id', f'unknown_{i}'),
                    'source': self._format_simple_source_name(source.get('source', 'Unknown')),
                    'page': source.get('page', 1),
                    'quote': self._format_quote(source.get('quote', '')),
                    'relevance_score': round(float(source.get('relevance_score', 0)), 2),
                    'similarity_score': round(float(source.get('similarity_score', 0)), 2)
                }
                
                # Add file type if available
                source_path = source.get('source', '')
                if source_path:
                    ext = os.path.splitext(source_path)[1].lower()
                    type_map = {'.pdf': 'PDF', '.docx': 'Word', '.txt': 'Text'}
                    formatted_source['file_type'] = type_map.get(ext, 'Document')
                
                formatted_sources.append(formatted_source)
                
            except Exception as e:
                logger.warning(f"Error formatting source {i}: {e}")
                continue
        
        return formatted_sources
    
    def _format_quote(self, quote: str) -> str:
        """Format quote text for display"""
        if not quote:
            return ""
        
        # Clean and truncate quote
        cleaned_quote = ' '.join(quote.split())
        
        max_quote_length = 150
        if len(cleaned_quote) > max_quote_length:
            # Try to truncate at sentence boundary
            truncated = cleaned_quote[:max_quote_length].rsplit('.', 1)[0]
            if len(truncated) > max_quote_length // 2:
                cleaned_quote = truncated + "..."
            else:
                cleaned_quote = cleaned_quote[:max_quote_length - 3] + "..."
        
        return cleaned_quote
    
    def _validate_confidence(self, confidence: str) -> str:
        """Validate confidence level"""
        valid_confidence = ['high', 'medium', 'low']
        
        confidence = confidence.lower().strip() if confidence else 'medium'
        
        return confidence if confidence in valid_confidence else 'medium'
    
    def _create_metadata(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create metadata about the response"""
        metadata = {
            'total_sources': len(sources),
            'avg_similarity': self._calculate_avg_similarity(sources),
            'processing_info': {
                'rag_enabled': True,
                'pdf_processing': True,
                'source_attribution': len(sources) > 0
            }
        }
        
        if sources:
            metadata['source_types'] = self._get_source_types(sources)
        
        return metadata
    
    def _get_source_types(self, sources: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get count of different source types"""
        type_counts = {}
        
        for source in sources:
            source_path = source.get('source', '')
            
            if source_path:
                ext = os.path.splitext(source_path)[1].lower()
                
                type_map = {
                    '.pdf': 'PDF',
                    '.docx': 'Word Document', 
                    '.txt': 'Text File'
                }
                
                file_type = type_map.get(ext, 'Other')
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return type_counts
    
    def _calculate_avg_similarity(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate average similarity score"""
        if not sources:
            return 0.0
        
        similarity_scores = []
        for source in sources:
            score = source.get('similarity_score', 0)
            try:
                similarity_scores.append(float(score))
            except (ValueError, TypeError):
                continue
        
        return round(sum(similarity_scores) / len(similarity_scores), 2) if similarity_scores else 0.0
    
    def _generate_response_id(self) -> str:
        """Generate unique response ID"""
        return str(uuid.uuid4())[:8]
    
    def format_for_api(self, response: Dict[str, Any]) -> str:
        """Format response as JSON string for API"""
        try:
            return json.dumps(response, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error formatting response for API: {str(e)}")
            return json.dumps({
                'decision': 'answered',
                'justification': 'Error formatting response',
                'sources': [],
                'confidence': 'low'
            })
    
    def format_multiple_answers(self, answers: List[str], sources_list: Optional[List[List[Dict[str, Any]]]] = None) -> List[str]:
        """Format multiple answers for HackRx array response"""
        if not answers:
            return ["No answers available."]
        
        formatted_answers = []
        
        for i, answer in enumerate(answers):
            sources = sources_list[i] if sources_list and i < len(sources_list) else None
            formatted_answer = self.format_simple_answer(answer, sources)
            formatted_answers.append(formatted_answer)
        
        return formatted_answers
    
    def create_error_response(self, error_message: str) -> str:
        """Create formatted error response"""
        clean_error = str(error_message)[:200]  # Limit error message length
        return f"I apologize, but I encountered an issue while processing your request: {clean_error}"
    
    def enhance_answer_quality(self, answer: str, query: str) -> str:
        """Enhance answer quality based on query type"""
        if not answer or len(answer) < 20:
            return answer
        
        query_lower = query.lower()
        
        # Add context for specific question types
        if query_lower.startswith('what is'):
            # Ensure answer starts with a definition
            if not answer.lower().startswith(('is ', 'are ', 'means', 'refers to')):
                # Try to restructure as definition
                subject = query[7:].strip().rstrip('?')  # Remove "what is" and "?"
                if subject and len(subject) < 50:
                    answer = f"{subject.title()} is {answer.lower()}" if not answer[0].islower() else f"{subject.title()} {answer}"
        
        elif query_lower.startswith(('how much', 'what is the cost', 'what is the price')):
            # Emphasize amounts in cost-related queries
            if '$' in answer or '₹' in answer or any(word in answer.lower() for word in ['cost', 'price', 'fee', 'amount']):
                # Amount already mentioned, no change needed
                pass
            else:
                # Add qualifier if no amount found
                answer = f"Based on the available information: {answer}"
        
        return answer
    
    def get_stats(self) -> Dict[str, Any]:
        """Get formatter statistics and configuration"""
        return {
            'max_response_length': self.max_response_length,
            'standard_fields': self.standard_fields,
            'version': '2.0',
            'optimized_for': 'HackRx RAG System'
        }

# Create global instance
response_formatter = ResponseFormatter()
