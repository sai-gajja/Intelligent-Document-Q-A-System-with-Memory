# src/vector_db.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
import logging
from config import config

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collections = self._initialize_collections()
        
    def _initialize_collections(self):
        """Initialize ChromaDB collections"""
        collections = {}
        try:
            # Main document chunks collection
            collections['document_chunks'] = self.client.get_or_create_collection(
                name="document_chunks",
                metadata={"description": "Document chunks with embeddings"}
            )
            
            # User interactions collection
            collections['user_interactions'] = self.client.get_or_create_collection(
                name="user_interactions",
                metadata={"description": "User query and feedback history"}
            )
            
            # Feedback data collection
            collections['feedback_data'] = self.client.get_or_create_collection(
                name="feedback_data",
                metadata={"description": "Explicit and implicit feedback"}
            )
            
            # Q&A pairs collection
            collections['qa_pairs'] = self.client.get_or_create_collection(
                name="qa_pairs",
                metadata={"description": "Successful Q&A pairs"}
            )
            
            logger.info("All collections initialized successfully")
            return collections
            
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
    
    def store_document_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Store document chunks with embeddings"""
        try:
            ids = [chunk['chunk_id'] for chunk in chunks]
            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            
            self.collections['document_chunks'].add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"Stored {len(chunks)} document chunks")
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: List[float], n_results: int = 5, 
                            filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        try:
            results = self.collections['document_chunks'].query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            similar_chunks = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    similar_chunks.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'id': results['ids'][0][i]
                    })
                
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def store_user_interaction(self, session_id: str, query: str, answer: str, 
                             feedback: Optional[Dict] = None):
        """Store user interaction data"""
        try:
            interaction_id = str(uuid.uuid4())
            metadata = {
                'session_id': session_id,
                'query': query,
                'answer': answer,
                'timestamp': str(uuid.uuid4()),
                'feedback': feedback or {}
            }
            
            self.collections['user_interactions'].add(
                ids=[interaction_id],
                documents=[query + " " + answer],
                metadatas=[metadata]
            )
            
            return interaction_id
            
        except Exception as e:
            logger.error(f"Error storing user interaction: {e}")
            return str(uuid.uuid4())
    
    def store_feedback(self, interaction_id: str, feedback_type: str, 
                      feedback_data: Dict, corrected_answer: Optional[str] = None):
        """Store explicit and implicit feedback"""
        try:
            feedback_id = str(uuid.uuid4())
            metadata = {
                'interaction_id': interaction_id,
                'feedback_type': feedback_type,
                'feedback_data': feedback_data,
                'corrected_answer': corrected_answer,
                'timestamp': str(uuid.uuid4())
            }
            
            self.collections['feedback_data'].add(
                ids=[feedback_id],
                documents=[str(feedback_data)],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
    
    def store_qa_pair(self, question: str, answer: str, topic: str, confidence: float):
        """Store successful Q&A pairs for long-term memory"""
        try:
            qa_id = str(uuid.uuid4())
            metadata = {
                'question': question,
                'answer': answer,
                'topic': topic,
                'confidence': confidence,
                'usage_count': 1,
                'timestamp': str(uuid.uuid4())
            }
            
            self.collections['qa_pairs'].add(
                ids=[qa_id],
                documents=[question + " " + answer],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Error storing Q&A pair: {e}")
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            results = self.collections['user_interactions'].get(
                where={"session_id": session_id},
                limit=limit
            )
            
            history = []
            for i in range(len(results['ids'])):
                history.append({
                    'query': results['metadatas'][i]['query'],
                    'answer': results['metadatas'][i]['answer'],
                    'timestamp': results['metadatas'][i]['timestamp'],
                    'feedback': results['metadatas'][i]['feedback']
                })
                
            return sorted(history, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []