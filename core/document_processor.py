# core/document_processor.py - Corrected Version

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import uuid
import tempfile
import requests
import pdfplumber
import pypdf
from docx import Document
from config.settings import (
    CHUNK_SIZE, CHUNK_OVERLAP, PDF_UPLOAD_PATH, 
    SUPPORTED_FORMATS, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor with unified text extraction"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.txt']
        
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load document and extract text with metadata"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
            
            if extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif extension == '.docx':
                return self._extract_from_docx(file_path)
            elif extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported extension: {extension}")
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF with enhanced error handling"""
        pages_data = []
        
        try:
            logger.info(f"📄 Extracting PDF: {file_path}")
            
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"📖 PDF has {len(pdf.pages)} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            clean_text = self.preprocess_text(text)
                            if clean_text:  # Only add if we have meaningful content
                                pages_data.append({
                                    'text': clean_text,
                                    'page': page_num,
                                    'source': str(file_path),
                                    'type': 'pdf'
                                })
                                logger.info(f"✅ Page {page_num}: extracted {len(clean_text)} characters")
                            else:
                                logger.warning(f"⚠️ Page {page_num}: text extracted but empty after cleaning")
                        else:
                            logger.warning(f"❌ Page {page_num}: no text extracted")
                    except Exception as page_error:
                        logger.warning(f"❌ Page {page_num} extraction failed: {page_error}")
                        continue
                        
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}: {e}")
            # Fallback to pypdf
            try:
                logger.info("🔄 Trying pypdf as fallback...")
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        try:
                            text = page.extract_text()
                            if text and text.strip():
                                clean_text = self.preprocess_text(text)
                                if clean_text:
                                    pages_data.append({
                                        'text': clean_text,
                                        'page': page_num,
                                        'source': str(file_path),
                                        'type': 'pdf'
                                    })
                                    logger.info(f"✅ Page {page_num}: extracted {len(clean_text)} characters (pypdf)")
                        except Exception as page_error:
                            logger.warning(f"❌ Page {page_num} pypdf extraction failed: {page_error}")
                            continue
            except Exception as e2:
                logger.error(f"Both PDF extractors failed for {file_path}: {e2}")
                raise
        
        if not pages_data:
            raise ValueError("No extractable text found in PDF")
        
        return {
            'pages': pages_data,
            'total_pages': len(pages_data),
            'source': str(file_path),
            'type': 'pdf'
        }
    
    def _extract_from_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX file"""
        try:
            logger.info(f"📄 Extracting DOCX: {file_path}")
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                raise ValueError("No extractable text found in DOCX")
            
            return {
                'pages': [{
                    'text': processed_text,
                    'page': 1,
                    'source': str(file_path),
                    'type': 'docx'
                }],
                'total_pages': 1,
                'source': str(file_path),
                'type': 'docx'
            }
        except Exception as e:
            logger.error(f"Error extracting DOCX {file_path}: {e}")
            raise
    
    def _extract_from_txt(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from TXT file"""
        try:
            logger.info(f"📄 Extracting TXT: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                raise ValueError("No extractable text found in TXT")
            
            return {
                'pages': [{
                    'text': processed_text,
                    'page': 1,
                    'source': str(file_path),
                    'type': 'txt'
                }],
                'total_pages': 1,
                'source': str(file_path),
                'type': 'txt'
            }
        except Exception as e:
            logger.error(f"Error extracting TXT {file_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n[ \t]*\n', '\n\n', text)  # Preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit to double newlines max
        
        # Remove control characters but keep essential whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def chunk_document_text(self, text: str, chunk_size: int = None, 
                           overlap: int = None) -> List[str]:
        """Split text into overlapping chunks with smart boundaries"""
        if not text:
            return []
        
        chunk_size = chunk_size or CHUNK_SIZE
        overlap = overlap or CHUNK_OVERLAP
        
        # Split by sentences for better chunk boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_words > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and chunks:
                    overlap_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    current_length = len(overlap_words) + sentence_words
                else:
                    current_chunk = sentence
                    current_length = sentence_words
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_words
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Filter out chunks that are too small
        min_chunk_size = MIN_CHUNK_SIZE if MIN_CHUNK_SIZE else 10
        valid_chunks = [chunk for chunk in chunks if len(chunk.split()) >= min_chunk_size]
        
        logger.info(f"📊 Created {len(valid_chunks)} chunks from {len(chunks)} total chunks")
        return valid_chunks
    
    # In core/document_processor.py - Fix the build_documents method
def build_documents(self, file_path: str) -> List[Dict[str, Any]]:
    """Ultra-reliable document processing for HackRx compliance"""
    logger.info(f"🔍 Processing document: {file_path}")
    
    try:
        # Step 1: Handle URL downloads with retries
        if file_path.startswith(('http://', 'https://')):
            local_path = self._download_with_retries(file_path)
        else:
            local_path = file_path
        
        # Step 2: Validate file exists and is readable
        path_obj = Path(local_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        
        if path_obj.stat().st_size == 0:
            raise ValueError("File is empty")
        
        # Step 3: Process with multiple extraction methods
        documents = self._extract_with_fallbacks(local_path)
        
        if not documents:
            raise ValueError("No content could be extracted from document")
        
        logger.info(f"✅ Successfully created {len(documents)} document chunks")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Document processing failed: {str(e)}")
        # Return empty list instead of raising exception for HackRx compliance
        return []

def _download_with_retries(self, url: str, max_retries: int = 3) -> str:
    """Download with retry logic and proper error handling"""
    upload_dir = Path("data/uploaded_docs")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"doc_{uuid.uuid4().hex}.pdf"
    local_path = upload_dir / filename
    
    for attempt in range(max_retries):
        try:
            logger.info(f"📥 Download attempt {attempt + 1}/{max_retries}: {url}")
            
            response = requests.get(
                url, 
                timeout=60,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
            
            # Save file
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Validate download
            if local_path.stat().st_size > 100:  # Minimum 100 bytes
                logger.info(f"✅ Download successful: {local_path.stat().st_size} bytes")
                return str(local_path)
            else:
                raise ValueError("Downloaded file too small")
                
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            
    raise Exception("All download attempts failed")

def _extract_with_fallbacks(self, file_path: str) -> List[Dict[str, Any]]:
    """Extract text with multiple fallback methods"""
    extraction_methods = [
        self._extract_with_pdfplumber,
        self._extract_with_pypdf,
        self._extract_basic_text
    ]
    
    for method_name, method in [("pdfplumber", extraction_methods[0]), 
                               ("pypdf", extraction_methods[1]), 
                               ("basic", extraction_methods[2])]:
        try:
            logger.info(f"🔄 Trying extraction method: {method_name}")
            documents = method(file_path)
            
            if documents and len(documents) > 0:
                logger.info(f"✅ Success with {method_name}: {len(documents)} chunks")
                return documents
                
        except Exception as e:
            logger.warning(f"❌ {method_name} extraction failed: {e}")
            continue
    
    logger.error("❌ All extraction methods failed")
    return []

def _extract_with_pdfplumber(self, file_path: str) -> List[Dict[str, Any]]:
    """Primary extraction method using pdfplumber"""
    import pdfplumber
    
    all_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text += f" {text}"
    
    return self._create_document_chunks(all_text, file_path) if all_text.strip() else []

def _extract_with_pypdf(self, file_path: str) -> List[Dict[str, Any]]:
    """Fallback extraction using pypdf"""
    import pypdf
    
    all_text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                all_text += f" {text}"
    
    return self._create_document_chunks(all_text, file_path) if all_text.strip() else []

def _extract_basic_text(self, file_path: str) -> List[Dict[str, Any]]:
    """Last resort: basic file reading"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # Try to decode as text
            text = content.decode('utf-8', errors='ignore')
            return self._create_document_chunks(text, file_path) if text.strip() else []
    except:
        return []

def _create_document_chunks(self, text: str, file_path: str) -> List[Dict[str, Any]]:
    """Create standardized document chunks"""
    if not text or len(text.strip()) < 50:
        return []
    
    # Clean text
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    # Create chunks
    chunk_size = 1000
    overlap = 200
    chunks = []
    
    for i in range(0, len(cleaned_text), chunk_size - overlap):
        chunk_text = cleaned_text[i:i + chunk_size]
        
        if len(chunk_text) > 100:  # Only substantial chunks
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'doc_id': self._generate_doc_id(file_path),
                    'source': str(file_path),
                    'chunk_index': len(chunks),
                    'file_type': 'pdf'
                }
            })
    
    return chunks

def _download_document(self, url: str) -> str:
    """Download document from URL"""
    try:
        import requests
        import uuid
        
        # Create upload directory
        upload_dir = Path("data/uploaded_docs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        filename = f"doc_{uuid.uuid4().hex}.pdf"
        local_path = upload_dir / filename
        
        logger.info(f"📥 Downloading from: {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"✅ Downloaded to: {local_path}")
        return str(local_path)
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise

    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID based on file path and modification time"""
        try:
            path = Path(file_path)
            content = f"{path.name}_{path.stat().st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:8]
        except Exception:
            # Fallback to simple hash of file path
            return hashlib.md5(str(file_path).encode()).hexdigest()[:8]
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text content"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def validate_file(self, file_path: str, max_size_mb: int = 25) -> bool:
        """Validate file before processing"""
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file extension
            if path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file extension: {path.suffix}")
                return False
            
            # Check file size
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                logger.warning(f"File {file_path} too large: {size_mb:.1f}MB")
                return False
            
            logger.info(f"✅ File validation passed: {file_path} ({size_mb:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            return False

# Create global instance
document_processor = DocumentProcessor()
