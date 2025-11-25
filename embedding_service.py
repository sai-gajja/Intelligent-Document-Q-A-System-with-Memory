# src/embedding_service.py
import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
from config import config
import time
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.embedding_model = config.EMBEDDING_MODEL
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts"""
        embeddings = []
        
        for text in texts:
            try:
                # Gemini embedding generation
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Fallback: generate zero vector
                embeddings.append([0.0] * 768)
                
        return embeddings
    
    def generate_hierarchical_embeddings(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate hierarchical embeddings for document structure"""
        document_level_text = " ".join([chunk['content'] for chunk in chunks])
        
        # Document-level embedding
        doc_embedding = self.generate_embeddings([document_level_text])[0]
        
        # Section-level embeddings (group chunks)
        section_embeddings = {}
        sections = self._group_into_sections(chunks)
        
        for section_id, section_chunks in sections.items():
            section_text = " ".join([chunk['content'] for chunk in section_chunks])
            section_embedding = self.generate_embeddings([section_text])[0]
            section_embeddings[section_id] = section_embedding
        
        # Chunk-level embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        chunk_embeddings = self.generate_embeddings(chunk_texts)
        
        return {
            'document': doc_embedding,
            'sections': section_embeddings,
            'chunks': chunk_embeddings
        }
    
    def _group_into_sections(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group chunks into logical sections"""
        sections = {}
        current_section = "section_1"
        sections[current_section] = []
        
        for chunk in chunks:
            # Simple section detection based on headings
            content = chunk['content']
            if any(marker in content.lower() for marker in ['##', 'introduction', 'method', 'result', 'conclusion']):
                current_section = f"section_{len(sections) + 1}"
                sections[current_section] = []
            
            sections[current_section].append(chunk)
            
        return sections
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0.0