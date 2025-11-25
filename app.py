# frontend/app.py
import streamlit as st
import requests
import uuid
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

def init_session():
    """Initialize session state"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = []

def main():
    st.set_page_config(
        page_title="Intelligent Document Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Intelligent Document Q&A System")
    st.markdown("Upload documents and ask questions with memory and learning capabilities!")
    
    init_session()
    
    # Sidebar for document upload and system info
    with st.sidebar:
        st.header("Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'txt', 'html', 'md'],
            help="Supported formats: PDF, DOCX, TXT, HTML, MD"
        )
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_BASE_URL}/upload-document", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.uploaded_documents.append({
                            'name': uploaded_file.name,
                            'id': result['document_id'],
                            'chunks': result['chunks_processed']
                        })
                        st.success(f"Document processed! {result['chunks_processed']} chunks created.")
                    else:
                        st.error("Error processing document")
        
        # Show uploaded documents
        if st.session_state.uploaded_documents:
            st.subheader("Uploaded Documents")
            for doc in st.session_state.uploaded_documents:
                st.write(f"📄 {doc['name']} (ID: {doc['id'][:8]}...)")
        
        # System metrics
        st.header("System Info")
        try:
            metrics_response = requests.get(f"{API_BASE_URL}/metrics")
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                st.metric("Documents", metrics['documents_processed'])
                st.metric("Total Interactions", metrics['total_interactions'])
                st.metric("Active Sessions", metrics['active_sessions'])
        except:
            st.info("Metrics unavailable")
    
    # Main layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat Interface")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for i, exchange in enumerate(st.session_state.conversation_history):
                with st.chat_message("user"):
                    st.write(exchange['query'])
                with st.chat_message("assistant"):
                    st.write(exchange['answer'])
                    
                    # Show confidence and feedback options
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    with col_a:
                        st.caption(f"Confidence: {exchange['confidence']:.2f}")
                    with col_b:
                        if st.button("👍", key=f"thumb_up_{i}"):
                            # Send positive feedback
                            feedback_data = {'rating': 5}
                            requests.post(f"{API_BASE_URL}/feedback", json={
                                'interaction_id': exchange.get('interaction_id', 'unknown'),
                                'feedback_type': 'rating',
                                'feedback_data': feedback_data
                            })
                            st.success("Thanks for your feedback!")
                    with col_c:
                        if st.button("👎", key=f"thumb_down_{i}"):
                            # Send negative feedback
                            feedback_data = {'rating': 1}
                            requests.post(f"{API_BASE_URL}/feedback", json={
                                'interaction_id': exchange.get('interaction_id', 'unknown'),
                                'feedback_type': 'rating',
                                'feedback_data': feedback_data
                            })
                            st.info("Thanks for your feedback! Consider providing a correction.")
    
    with col2:
        st.subheader("Memory & Learning")
        
        # Session info
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        
        # Conversation stats
        st.metric("Messages in session", len(st.session_state.conversation_history))
        
        # Memory visualization
        if st.button("View Conversation History"):
            try:
                history_response = requests.get(
                    f"{API_BASE_URL}/conversation-history/{st.session_state.session_id}"
                )
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    st.write("**Complete History:**")
                    for item in history_data['history'][-5:]:  # Last 5 items
                        st.text_area(
                            f"Q: {item['query']}",
                            f"A: {item['answer']}",
                            height=100,
                            key=f"history_{item.get('interaction_id', i)}"
                        )
            except:
                st.info("History unavailable")
        
        # Learning trigger
        if st.button("Trigger Learning from Feedback"):
            with st.spinner("Processing feedback..."):
                try:
                    learn_response = requests.post(f"{API_BASE_URL}/learn-from-feedback")
                    if learn_response.status_code == 200:
                        st.success("Learning process started!")
                    else:
                        st.error("Error triggering learning")
                except:
                    st.error("Cannot connect to learning service")

    # Chat input - MOVED OUTSIDE OF COLUMNS
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message to history
        st.session_state.conversation_history.append({
            'query': user_query,
            'answer': '',
            'timestamp': datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/query", json={
                        'query': user_query,
                        'session_id': st.session_state.session_id
                    })
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.write(result['answer'])
                        
                        # Display sources if available
                        if result['sources']:
                            with st.expander("View Sources"):
                                for source in result['sources']:
                                    st.write(f"Document: {source.get('doc_id', 'Unknown')}")
                                    st.write(f"Page: {source.get('page_num', 'N/A')}")
                                    st.caption(source.get('content', '')[:200] + "...")
                        
                        # Update conversation history
                        st.session_state.conversation_history[-1].update({
                            'answer': result['answer'],
                            'confidence': result['confidence'],
                            'sources': result['sources'],
                            'processing_time': result['processing_time']
                        })
                        
                        # Show performance info
                        st.caption(f"Confidence: {result['confidence']:.2f} | "
                                 f"Processing time: {result['processing_time']:.2f}s")
                        
                    else:
                        st.error("Error getting response from server")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()