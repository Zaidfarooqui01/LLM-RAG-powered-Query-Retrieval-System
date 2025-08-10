#!/bin/bash
# HackRx 6.0 RAG System Startup Script - Enhanced Version

set -e  # Exit on any error

echo "🚀 Starting HackRx 6.0 Intelligent Query-Retrieval System"
echo "============================================================="
echo "📅 Started at: $(date)"
echo ""

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found! Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
print_status "Python $PYTHON_VERSION detected"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    print_error "main.py not found! Please run this script from the project root directory."
    exit 1
fi

print_status "Project directory validated"

# Optional: Check OpenAI API key (only warn, don't require)
if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "OPENAI_API_KEY not set (optional - your system uses sentence-transformers)"
    print_info "If you plan to use GPT later, set: export OPENAI_API_KEY=your_key_here"
else
    print_status "OpenAI API key detected"
fi

# Create and activate virtual environment
echo ""
echo "📦 Setting up Python environment..."
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found! Creating basic requirements..."
    cat > requirements.txt << EOF
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pdfplumber==0.9.0
pypdf==3.0.1
python-docx==1.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
httpx==0.25.2
requests==2.31.0
python-dotenv==1.0.0
pathlib2==2.3.7
EOF
fi

# Install/update dependencies
echo ""
echo "📚 Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
print_status "Dependencies installed"

# Create all necessary directories
echo ""
echo "📁 Creating project directories..."
directories=(
    "data/uploaded_docs"
    "data/vector_store"
    "data/cache"
    "data/cache/embeddings"
    "data/cache/queries"  
    "data/cache/documents"
    "logs"
    "core"
    "utils"
    "config"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done
print_status "All directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env configuration file..."
    cat > .env << EOF
# HackRx 6.0 RAG System Configuration

# Core RAG Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_BACKEND=faiss

# File Processing
PDF_UPLOAD_PATH=data/uploaded_docs
VECTOR_STORE_PATH=data/vector_store
LOGS_PATH=logs
MAX_FILE_SIZE_MB=25

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
HACKRX_TOKEN=6ca800c46dd70bb4a8ef18a01692ac76721bb2b50303e31dbed18a186993ac1e

# Optional: OpenAI (for future use)
# OPENAI_API_KEY=your-key-here

# Performance
ENABLE_CACHING=true
ENABLE_DEBUG_LOGGING=true
LOG_LEVEL=INFO
EOF
    print_status "Configuration file created"
else
    print_status "Configuration file exists"
fi

# Validate core modules exist
echo ""
echo "🔍 Validating RAG system components..."
required_files=(
    "main.py"
    "config/settings.py"
    "core/document_processor.py"
    "core/embeddings_manager.py"
    "core/vector_store.py"
    "core/llm_manager.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists"
    else
        print_error "$file missing"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "Missing required files: ${missing_files[*]}"
    print_info "Please ensure all RAG components are in place before starting."
    exit 1
fi

# Test Python imports
echo ""
echo "🧪 Testing Python module imports..."
python3 -c "
try:
    print('Testing imports...')
    import fastapi
    print('✅ FastAPI available')
    import pdfplumber
    print('✅ PDF processing available')
    import sentence_transformers
    print('✅ Sentence transformers available')
    import faiss
    print('✅ FAISS available')
    import numpy
    print('✅ NumPy available')
    
    # Test our modules
    from config.settings import settings
    print('✅ Settings module loaded')
    from core.embeddings_manager import embeddings_manager
    print('✅ Embeddings manager loaded')
    print('🎉 All imports successful!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Module error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_status "All Python modules validated"
else
    print_error "Python module validation failed"
    exit 1
fi

# Quick system health check
echo ""
echo "🏥 Performing system health check..."
python3 -c "
try:
    from core.embeddings_manager import embeddings_manager
    test_embedding = embeddings_manager.embed_query('test')
    print(f'✅ Embedding system working (dimension: {test_embedding.shape[0]})')
    
    from core.vector_store import vector_store
    stats = vector_store.get_stats()
    print(f'✅ Vector store ready (vectors: {stats.get(\"total_vectors\", 0)})')
    
    print('🎉 Health check passed!')
except Exception as e:
    print(f'⚠️  Health check warning: {e}')
    print('System may still work, but check logs for issues.')
"

# Show system information
echo ""
echo "📊 SYSTEM INFORMATION"
echo "====================="
echo "🔧 Python Version: $PYTHON_VERSION"
echo "📁 Project Directory: $(pwd)"
echo "🌐 API Endpoint: http://localhost:8000/api/v1/hackrx/run"
echo "📚 Documentation: http://localhost:8000/api/v1/docs"
echo "🔍 Health Check: http://localhost:8000/api/v1/health"
echo "🐛 Debug Vector Store: http://localhost:8000/debug-vector-store/"
echo "💾 Debug Cache: http://localhost:8000/debug-cache/"
echo ""

# Final confirmation
echo "🎯 READY TO START!"
echo "=================="
print_info "Press Ctrl+C to stop the server"
print_info "Logs will appear below..."
echo ""

# Start the server with better configuration
echo "🌟 Starting HackRx RAG API server..."
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info \
    --access-log \
    --use-colors
