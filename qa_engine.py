# src/qa_engine.py
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import time
import logging
from .embedding_service import EmbeddingService
from .vector_db import VectorDatabase
from .memory_system import MemorySystem
from config import config

logger = logging.getLogger(__name__)

class QAEngine:
    def __init__(self, embedding_service: EmbeddingService, 
                 vector_db: VectorDatabase, memory_system: MemorySystem):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.memory_system = memory_system
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # Cache for frequently asked questions
        self.qa_cache: Dict[str, Dict] = {}
        
    # src/qa_engine.py - Fix the process_query method
    def process_query(self, query: str, session_id: str, 
                     document_filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Process user query and generate answer"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, session_id)
            if cache_key in self.qa_cache:
                logger.info("Cache hit for query")
                return self.qa_cache[cache_key]
            
            # Step 1: Query expansion using conversation context
            expanded_query = self._expand_query(query, session_id)
            
            # Step 2: Generate query embedding
            query_embedding = self.embedding_service.generate_embeddings([expanded_query])[0]
            
            # Step 3: Retrieve relevant chunks
            relevant_chunks = self.vector_db.search_similar_chunks(
                query_embedding, 
                n_results=5,
                filters=document_filters
            )
            
            # Step 4: Search long-term memory for similar Q&A
            similar_qa = self.memory_system.search_long_term_memory(query)
            
            # Step 5: Prepare context
            context = self._prepare_context(relevant_chunks, similar_qa, session_id)
            
            # Step 6: Generate answer using Gemini
            answer = self._generate_answer(query, context, session_id)
            
            # Step 7: Store interaction and get interaction ID
            interaction_id = self.vector_db.store_user_interaction(session_id, query, answer)
            
            # Step 8: Add to short-term memory with interaction ID
            self.memory_system.add_to_short_term_memory(
                session_id, query, answer, 
                feedback={'interaction_id': interaction_id}
            )
            
            # Step 9: Calculate confidence
            confidence = self._calculate_confidence(answer, relevant_chunks)
            
            # Step 10: Cache the result
            result = {
                'answer': answer,
                'confidence': confidence,
                'sources': [chunk['metadata'] for chunk in relevant_chunks],
                'similar_qa': similar_qa,
                'processing_time': time.time() - start_time,
                'interaction_id': interaction_id
            }
            
            self.qa_cache[cache_key] = result
            
            # Step 11: Add to long-term memory if high confidence
            if confidence > 0.8:
                topic = self._extract_topic(query)
                self.memory_system.add_to_long_term_memory(query, answer, topic, confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': "I apologize, but I encountered an error processing your query. Please try again.",
                'confidence': 0.0,
                'sources': [],
                'similar_qa': [],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _expand_query(self, query: str, session_id: str) -> str:
        """Expand query using conversation context"""
        context = self.memory_system.get_short_term_context(session_id)
        
        if not context:
            return query
        
        # Use Gemini to reformulate query based on context
        context_text = "\n".join([
            f"Q: {item['query']}\nA: {item['answer']}" 
            for item in context[-3:]  # Last 3 exchanges
        ])
        
        prompt = f"""
        Based on the following conversation context, reformulate the current query to make it more clear and contextual.
        
        Conversation Context:
        {context_text}
        
        Current Query: {query}
        
        Reformulated Query:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    def _prepare_context(self, chunks: List[Dict], similar_qa: List[Dict], 
                        session_id: str) -> str:
        """Prepare context for answer generation"""
        context_parts = []
        
        # Add document chunks
        if chunks:
            context_parts.append("Relevant Document Content:")
            for i, chunk in enumerate(chunks):
                context_parts.append(f"[Source {i+1}]: {chunk['content']}")
        
        # Add similar Q&A from memory
        if similar_qa:
            context_parts.append("\nRelated Previous Questions and Answers:")
            for i, qa in enumerate(similar_qa):
                context_parts.append(f"Q: {qa['question']}")
                context_parts.append(f"A: {qa['answer']}")
        
        # Add conversation context
        short_term_context = self.memory_system.get_short_term_context(session_id)
        if short_term_context:
            context_parts.append("\nRecent Conversation:")
            for item in short_term_context[-2:]:  # Last 2 exchanges
                context_parts.append(f"User: {item['query']}")
                context_parts.append(f"Assistant: {item['answer']}")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, session_id: str) -> str:
        """Generate answer using Gemini"""
        prompt = f"""
        You are an intelligent document Q&A assistant. Use the following context to answer the user's question.
        
        Context:
        {context}
        
        User Question: {query}
        
        Instructions:
        1. Answer based only on the provided context
        2. If the context doesn't contain the answer, say "I cannot find the answer in the provided documents"
        3. Be concise and accurate
        4. Cite sources when relevant
        5. Maintain conversation flow considering recent exchanges
        
        Answer:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text if response.text else "I cannot generate an answer at the moment."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I'm having trouble generating an answer right now."
    
    def _calculate_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """Calculate confidence score for the answer"""
        if not chunks:
            return 0.0
        
        # Simple confidence calculation based on answer length and chunk relevance
        base_confidence = min(len(answer) / 100, 1.0)  # Longer answers might be more confident
        
        # Adjust based on chunk distances (lower distance = higher confidence)
        avg_distance = sum(chunk['distance'] for chunk in chunks) / len(chunks)
        distance_confidence = max(0, 1 - avg_distance)
        
        return (base_confidence + distance_confidence) / 2
    
    def _extract_topic(self, query: str) -> str:
        """Extract topic from query for memory organization"""
        # Simple topic extraction - in practice, use more sophisticated NLP
        topics = ['technology', 'science', 'history', 'business', 'health', 'education']
        
        query_lower = query.lower()
        for topic in topics:
            if topic in query_lower:
                return topic
        
        return 'general'
    
    def _generate_cache_key(self, query: str, session_id: str) -> str:
        """Generate cache key for query"""
        import hashlib
        content = f"{query}_{session_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def provide_feedback(self, interaction_id: str, feedback_type: str, 
                        feedback_data: Dict, corrected_answer: Optional[str] = None):
        """Process user feedback"""
        self.vector_db.store_feedback(interaction_id, feedback_type, feedback_data, corrected_answer)
        
        # If correction provided, add to learning pipeline
        if corrected_answer:
            self._learn_from_correction(interaction_id, corrected_answer)
    
    def _learn_from_correction(self, interaction_id: str, corrected_answer: str):
        """Learn from user corrections"""
        # This would trigger the learning pipeline
        # For now, we just log the correction
        logger.info(f"Learning from correction for interaction {interaction_id}")