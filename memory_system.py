# src/memory_system.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    content: Any
    timestamp: float
    memory_type: str
    metadata: Dict[str, Any]

class MemorySystem:
    def __init__(self, vector_db, short_term_size: int = 20):
        self.vector_db = vector_db
        self.short_term_size = short_term_size
        self.short_term_memory: List[MemoryItem] = []
        self.sessions: Dict[str, List[MemoryItem]] = {}
        
    def add_to_short_term_memory(self, session_id: str, query: str, answer: str, 
                               feedback: Optional[Dict] = None):
        """Add interaction to short-term memory"""
        memory_item = MemoryItem(
            content={'query': query, 'answer': answer, 'feedback': feedback},
            timestamp=time.time(),
            memory_type='short_term',
            metadata={'session_id': session_id}
        )
        
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append(memory_item)
        
        if len(self.sessions[session_id]) > self.short_term_size:
            self.sessions[session_id] = self.sessions[session_id][-self.short_term_size:]
        
        self.short_term_memory.append(memory_item)
        if len(self.short_term_memory) > self.short_term_size:
            self.short_term_memory = self.short_term_memory[-self.short_term_size:]
    
    def get_short_term_context(self, session_id: str) -> List[Dict]:
        """Get recent conversation context for a session"""
        if session_id not in self.sessions:
            return []
        
        recent_interactions = self.sessions[session_id][-self.short_term_size:]
        context = []
        
        for item in recent_interactions:
            context.append(item.content)
            
        return context
    
    def add_to_long_term_memory(self, question: str, answer: str, topic: str, 
                              confidence: float):
        """Add successful Q&A to long-term memory"""
        try:
            self.vector_db.store_qa_pair(question, answer, topic, confidence)
            logger.info(f"Added Q&A to long-term memory: {topic}")
            
        except Exception as e:
            logger.error(f"Error adding to long-term memory: {e}")
    
    def search_long_term_memory(self, query: str, topic: Optional[str] = None, 
                              limit: int = 3) -> List[Dict]:
        """Search long-term memory for relevant Q&A pairs"""
        try:
            # Simple implementation - in production, use semantic search
            qa_collection = self.vector_db.collections['qa_pairs']
            
            filters = {}
            if topic:
                filters = {"topic": topic}
            
            results = qa_collection.get(
                where=filters,
                limit=limit
            )
            
            qa_pairs = []
            for i in range(len(results['ids'])):
                qa_pairs.append({
                    'question': results['metadatas'][i]['question'],
                    'answer': results['metadatas'][i]['answer'],
                    'topic': results['metadatas'][i]['topic'],
                    'confidence': results['metadatas'][i]['confidence'],
                    'usage_count': results['metadatas'][i]['usage_count']
                })
                
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error searching long-term memory: {e}")
            return []
    
    def get_episodic_memory(self, session_id: str) -> List[Dict]:
        """Get complete interaction history for a session"""
        return self.vector_db.get_conversation_history(session_id)
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600):
        """Clean up old sessions from memory"""
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, memory_items in self.sessions.items():
            if memory_items and (current_time - memory_items[0].timestamp) > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")