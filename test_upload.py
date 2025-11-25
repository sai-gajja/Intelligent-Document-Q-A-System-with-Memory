# test_fix.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.vector_db import VectorDatabase
from src.document_processor import DocumentProcessor
from src.embedding_service import EmbeddingService

def test_system():
    print("Testing system components...")
    
    try:
        # Test vector DB
        print("1. Testing Vector Database...")
        vector_db = VectorDatabase()
        print("✓ Vector DB initialized")
        
        # Test document processor
        print("2. Testing Document Processor...")
        processor = DocumentProcessor()
        print("✓ Document Processor initialized")
        
        # Test embedding service
        print("3. Testing Embedding Service...")
        embedding_service = EmbeddingService()
        test_embedding = embedding_service.generate_embeddings(["test document"])
        print(f"✓ Embedding Service working - generated vector of length: {len(test_embedding[0])}")
        
        print("\n🎉 All components working! System should now be functional.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()