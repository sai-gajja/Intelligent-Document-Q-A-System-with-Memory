# src/learning_pipeline.py
import logging
from typing import Dict, List, Any
import json
import os
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class LearningPipeline:
    def __init__(self, vector_db, qa_engine):
        self.vector_db = vector_db
        self.qa_engine = qa_engine
        self.feedback_storage_path = config.FEEDBACK_STORAGE_PATH
        
        # Create feedback directory if it doesn't exist
        os.makedirs(self.feedback_storage_path, exist_ok=True)
    
    def process_feedback_batch(self):
        """Process accumulated feedback for learning"""
        try:
            feedback_collection = self.vector_db.client.get_collection("feedback_data")
            feedback_data = feedback_collection.get()
            
            corrections = []
            ratings = []
            
            for i in range(len(feedback_data['ids'])):
                metadata = feedback_data['metadatas'][i]
                feedback_type = metadata['feedback_type']
                
                if feedback_type == 'correction' and metadata['corrected_answer']:
                    corrections.append({
                        'interaction_id': metadata['interaction_id'],
                        'original_query': self._get_original_query(metadata['interaction_id']),
                        'corrected_answer': metadata['corrected_answer'],
                        'timestamp': metadata['timestamp']
                    })
                elif feedback_type == 'rating':
                    ratings.append({
                        'interaction_id': metadata['interaction_id'],
                        'rating': metadata['feedback_data'].get('rating', 0),
                        'timestamp': metadata['timestamp']
                    })
            
            # Apply learning from corrections
            if corrections:
                self._update_retrieval_weights(corrections)
                
            # Update answer strategies based on ratings
            if ratings:
                self._optimize_answer_strategies(ratings)
                
            # Save processed feedback
            self._archive_feedback(feedback_data)
            
            logger.info(f"Processed {len(corrections)} corrections and {len(ratings)} ratings")
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
    
    def _get_original_query(self, interaction_id: str) -> str:
        """Get original query for an interaction"""
        try:
            interactions_collection = self.vector_db.client.get_collection("user_interactions")
            results = interactions_collection.get(ids=[interaction_id])
            
            if results['metadatas']:
                return results['metadatas'][0]['query']
            return ""
            
        except Exception as e:
            logger.error(f"Error getting original query: {e}")
            return ""
    
    def _update_retrieval_weights(self, corrections: List[Dict]):
        """Update retrieval weights based on corrections"""
        # This would implement more sophisticated retrieval optimization
        # For now, we log the corrections for analysis
        
        correction_file = os.path.join(
            self.feedback_storage_path, 
            f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(correction_file, 'w') as f:
            json.dump(corrections, f, indent=2)
        
        logger.info(f"Saved {len(corrections)} corrections to {correction_file}")
    
    def _optimize_answer_strategies(self, ratings: List[Dict]):
        """Optimize answer strategies based on user ratings"""
        high_rated = [r for r in ratings if r['rating'] >= 4]
        low_rated = [r for r in ratings if r['rating'] <= 2]
        
        # Analyze patterns in high vs low rated answers
        patterns_file = os.path.join(
            self.feedback_storage_path,
            f"rating_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        analysis = {
            'high_rated_count': len(high_rated),
            'low_rated_count': len(low_rated),
            'high_rated_examples': high_rated[:5],  # Sample
            'low_rated_examples': low_rated[:5]     # Sample
        }
        
        with open(patterns_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Rating analysis saved to {patterns_file}")
    
    def _archive_feedback(self, feedback_data: Dict):
        """Archive processed feedback"""
        archive_file = os.path.join(
            self.feedback_storage_path,
            f"processed_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(archive_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        # Clear processed feedback from database
        try:
            feedback_collection = self.vector_db.client.get_collection("feedback_data")
            feedback_collection.delete(ids=feedback_data['ids'])
        except Exception as e:
            logger.error(f"Error clearing processed feedback: {e}")