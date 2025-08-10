# debug_processor.py - Enhanced Document Processor Diagnostic Tool

import sys
import traceback
from pathlib import Path

def debug_document_processor():
    """Comprehensive document processor debugging"""
    print("🔍 DOCUMENT PROCESSOR DIAGNOSTIC TOOL")
    print("=" * 50)
    
    try:
        # Test 1: Import check
        print("\n1. 📦 Testing import...")
        from core.document_processor import document_processor
        print("   ✅ Successfully imported document_processor")
        
        # Test 2: Check available methods
        print("\n2. 🔧 Available methods:")
        methods = [method for method in dir(document_processor) if not method.startswith('_')]
        for method in methods:
            print(f"   📝 {method}")
        
        # Test 3: Check required methods
        print("\n3. ✅ Required methods check:")
        required_methods = ['build_documents', 'load_document', 'preprocess_text']
        
        for method in required_methods:
            if hasattr(document_processor, method):
                print(f"   ✅ {method} method exists")
            else:
                print(f"   ❌ {method} method missing")
        
        # Test 4: Check legacy/optional methods
        print("\n4. 🔄 Optional/Legacy methods:")
        optional_methods = ['process_pdf', 'chunk_document_text', 'validate_file']
        
        for method in optional_methods:
            if hasattr(document_processor, method):
                print(f"   ✅ {method} method exists")
            else:
                print(f"   ⚠️  {method} method not found (optional)")
        
        # Test 5: Check supported file types
        print("\n5. 📄 File type support:")
        if hasattr(document_processor, 'supported_extensions'):
            extensions = document_processor.supported_extensions
            print(f"   📋 Supported extensions: {extensions}")
        else:
            print("   ⚠️  supported_extensions attribute not found")
        
        # Test 6: Test basic functionality
        print("\n6. 🧪 Basic functionality test:")
        test_text = "This is a test document for processing."
        
        try:
            if hasattr(document_processor, 'preprocess_text'):
                processed = document_processor.preprocess_text(test_text)
                print(f"   ✅ preprocess_text works: '{processed[:50]}...'")
            else:
                print("   ⚠️  Cannot test preprocess_text (method missing)")
        except Exception as e:
            print(f"   ❌ preprocess_text failed: {e}")
        
        # Test 7: Test with sample file path (if exists)
        print("\n7. 📁 Sample file processing test:")
        sample_files = [
            "data/uploaded_docs/sample.pdf",
            "sample.pdf",
            "test.pdf"
        ]
        
        sample_found = False
        for sample_file in sample_files:
            if Path(sample_file).exists():
                sample_found = True
                print(f"   📄 Found sample file: {sample_file}")
                
                try:
                    if hasattr(document_processor, 'build_documents'):
                        docs = document_processor.build_documents(sample_file)
                        print(f"   ✅ build_documents works: {len(docs)} chunks created")
                        if docs:
                            print(f"   📝 Sample chunk: '{docs[0].get('text', '')[:50]}...'")
                    else:
                        print("   ⚠️  Cannot test build_documents (method missing)")
                except Exception as e:
                    print(f"   ❌ build_documents failed: {e}")
                break
        
        if not sample_found:
            print("   ℹ️  No sample files found for testing")
        
        # Test 8: Configuration check
        print("\n8. ⚙️ Configuration check:")
        try:
            from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, PDF_UPLOAD_PATH
            print(f"   📊 Chunk size: {CHUNK_SIZE}")
            print(f"   🔄 Chunk overlap: {CHUNK_OVERLAP}")
            print(f"   📁 Upload path: {PDF_UPLOAD_PATH}")
            print(f"   📂 Upload path exists: {Path(PDF_UPLOAD_PATH).exists()}")
        except ImportError as e:
            print(f"   ❌ Configuration import failed: {e}")
        
        print("\n" + "=" * 50)
        print("🎯 DIAGNOSTIC SUMMARY:")
        
        # Check core functionality
        has_build_documents = hasattr(document_processor, 'build_documents')
        has_load_document = hasattr(document_processor, 'load_document')
        has_preprocess = hasattr(document_processor, 'preprocess_text')
        
        if has_build_documents and has_load_document:
            print("   ✅ READY: Document processor has core functionality")
        elif has_build_documents:
            print("   ⚠️  PARTIAL: Has build_documents but missing some methods")
        else:
            print("   ❌ NOT READY: Missing critical build_documents method")
        
        return has_build_documents
        
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {e}")
        print("   💡 Make sure you're running from the project root directory")
        print("   💡 Check if core/document_processor.py exists")
        return False
    
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def check_dependencies():
    """Check required dependencies for document processing"""
    print("\n📦 DEPENDENCY CHECK:")
    
    dependencies = [
        ('pdfplumber', 'PDF text extraction'),
        ('pypdf', 'PDF fallback processing'),
        ('python-docx', 'DOCX file processing'),
        ('pathlib', 'File path handling'),
        ('hashlib', 'Document ID generation'),
        ('re', 'Text preprocessing')
    ]
    
    missing_deps = []
    
    for dep, description in dependencies:
        try:
            if dep == 'python-docx':
                import docx
                module_name = 'docx'
            else:
                module_name = dep
                __import__(module_name)
            print(f"   ✅ {dep}: Available ({description})")
        except ImportError:
            print(f"   ❌ {dep}: Missing ({description})")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n💡 INSTALL MISSING DEPENDENCIES:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    else:
        print(f"   🎉 All dependencies available!")
        return True

def main():
    """Run complete diagnostic"""
    try:
        # Check dependencies first
        deps_ok = check_dependencies()
        
        # Run processor diagnostic
        processor_ok = debug_document_processor()
        
        print("\n" + "=" * 50)
        print("🏁 FINAL RESULT:")
        
        if deps_ok and processor_ok:
            print("   🎉 SUCCESS: Document processor is ready!")
            print("   ✅ All required methods are available")
            print("   ✅ All dependencies are installed")
        elif processor_ok:
            print("   ⚠️  PARTIAL: Processor methods OK, but missing dependencies")
        elif deps_ok:
            print("   ⚠️  PARTIAL: Dependencies OK, but processor has issues")
        else:
            print("   ❌ ISSUES: Both processor and dependencies need attention")
        
        return processor_ok and deps_ok
        
    except KeyboardInterrupt:
        print("\n⏹️  Diagnostic interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
