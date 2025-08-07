import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import logging

# Import utilities
try:
    from utils import (
        check_api_key, validate_pdf_file, format_file_size, 
        display_error_info, ProgressTracker, check_streamlit_cloud
    )
except ImportError:
    # Fallback if utils.py is not available
    def check_api_key():
        return bool(os.getenv('GEMINI_API_KEY'))
    def validate_pdf_file(file):
        return file is not None
    def format_file_size(size):
        return f"{size / (1024*1024):.1f}MB"
    def display_error_info(error, context=""):
        st.error(f"Error: {str(error)}")
    class ProgressTracker:
        def __init__(self, *args, **kwargs):
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
        def update(self, step, message=""):
            pass
        def complete(self, message=""):
            pass
        def cleanup(self):
            pass
    def check_streamlit_cloud():
        return False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for async event loop in Streamlit
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'document_chunks' not in st.session_state:
        st.session_state.document_chunks = []

initialize_session_state()

# Check API key before proceeding
if not check_api_key():
    st.stop()

# Initialize the LLM model
@st.cache_resource
def get_llm_model():
    """Initialize and cache the LLM model"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
    except Exception as e:
        display_error_info(e, "LLM model initialization")
        st.stop()

model = get_llm_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Document Assistant</h1>
    <p>Upload your PDF documents and chat with them using advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Show welcome message for new users
if not st.session_state.processed_files and not st.session_state.chat_history:
    st.markdown("""
    ### 👋 Welcome to AI Document Assistant!
    
    This powerful tool allows you to:
    - 📁 **Upload multiple PDF documents** at once
    - 🤖 **Chat with your documents** using natural language
    - 🔍 **Get source citations** for every answer
    - 💾 **Maintain conversation history** throughout your session
    
    **Getting Started:**
    1. Use the sidebar to upload your PDF files
    2. Click "🚀 Process Documents" to analyze them
    3. Start asking questions about your documents!
    
    **Sample Questions to Try:**
    - "What is the main topic of this document?"
    - "Can you summarize the key points?"
    - "What are the important dates mentioned?"
    - "Who are the main people discussed?"
    """)
    
    # Show deployment info if on Streamlit Cloud
    if check_streamlit_cloud():
        st.info("🌟 You're using the cloud version! Your documents are processed securely and not stored permanently.")
    
    st.markdown("---")

# Utility functions
def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content to track processed files"""
    return hashlib.md5(file_content).hexdigest()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file with error handling"""
    try:
        if not validate_pdf_file(pdf_file):
            return ""
            
        pdf_reader = PdfReader(pdf_file)
        text = ""
        total_pages = len(pdf_reader.pages)
        
        # Show progress for large PDFs
        if total_pages > 10:
            progress = ProgressTracker(total_pages, "Extracting text from PDF")
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text += page.extract_text() + "\n"
                if total_pages > 10:
                    progress.update(page_num + 1, f"Page {page_num + 1}/{total_pages}")
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        if total_pages > 10:
            progress.complete("Text extraction complete!")
            time.sleep(1)
            progress.cleanup()
            
        return text
    except Exception as e:
        display_error_info(e, "PDF text extraction")
        return ""

def create_text_chunks(text: str) -> List[str]:
    """Split text into chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_embeddings():
    """Get cached embeddings model"""
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv('GEMINI_API_KEY')
    )

def create_vector_store(chunks: List[str], file_hash: str) -> tuple:
    """Create vector store from text chunks"""
    try:
        embeddings = get_embeddings()
        
        # Process in batches to avoid rate limits
        batch_size = 8  # Reduced for better stability
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        progress = ProgressTracker(total_batches, "Creating embeddings")
        
        # Create persistent directory
        persist_directory = f"chroma_db_{file_hash}"
        
        # Create ChromaDB vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Add documents in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            progress.update(batch_num, f"Batch {batch_num}/{total_batches}")
            
            try:
                # Create metadata for each chunk
                metadatas = [{"chunk_id": i + j, "source": f"chunk_{i + j}"} for j in range(len(batch))]
                
                # Add texts to vector store
                vector_store.add_texts(
                    texts=batch,
                    metadatas=metadatas
                )
                
                # Rate limiting - more conservative
                if batch_num < total_batches:
                    time.sleep(2)
                    
            except Exception as e:
                display_error_info(e, f"Creating embeddings for batch {batch_num}")
                return None, None
        
        progress.complete("✅ Vector store created successfully!")
        time.sleep(1)
        progress.cleanup()
        
        return vector_store, persist_directory
        
    except Exception as e:
        display_error_info(e, "vector store creation")
        return None, None

def load_vector_store(persist_directory: str) -> Chroma:
    """Load existing vector store"""
    try:
        embeddings = get_embeddings()
        return Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        st.error(f"❌ Error loading vector store: {str(e)}")
        return None

def create_conversation_chain(vector_store: Chroma):
    """Create conversation chain with memory"""
    try:
        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"❌ Error creating conversation chain: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with"
    )
    
    if uploaded_files:
        st.markdown("### 📊 Document Statistics")
        
        total_files = len(uploaded_files)
        total_size = sum([file.size for file in uploaded_files]) / (1024 * 1024)  # MB
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", total_files)
        with col2:
            st.metric("Size", f"{total_size:.1f} MB")
        
        # Process files
        if st.button("🚀 Process Documents", type="primary"):
            all_text = ""
            processed_files_info = []
            
            with st.spinner("Processing documents..."):
                for file in uploaded_files:
                    file_hash = get_file_hash(file.read())
                    file.seek(0)  # Reset file pointer
                    
                    if file_hash not in st.session_state.processed_files:
                        text = extract_text_from_pdf(file)
                        if text:
                            all_text += text + "\n\n"
                            st.session_state.processed_files[file_hash] = {
                                'name': file.name,
                                'processed_at': datetime.now().isoformat(),
                                'text_length': len(text)
                            }
                            processed_files_info.append(file.name)
                    else:
                        st.info(f"📄 {file.name} already processed")
            
            if all_text:
                # Create chunks
                chunks = create_text_chunks(all_text)
                st.session_state.document_chunks = chunks
                
                # Create vector store
                combined_hash = hashlib.md5(all_text.encode()).hexdigest()
                vector_store, index_path = create_vector_store(chunks, combined_hash)
                
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.conversation_chain = create_conversation_chain(vector_store)
                    
                    st.success(f"✅ Processed {len(processed_files_info)} documents successfully!")
                    st.balloons()
    
    # Document info
    if st.session_state.processed_files:
        st.markdown("### 📋 Processed Documents")
        for file_hash, info in st.session_state.processed_files.items():
            with st.expander(f"📄 {info['name']}"):
                st.write(f"**Processed:** {info['processed_at'][:19]}")
                st.write(f"**Text Length:** {info['text_length']:,} characters")
    
    # Session management
    st.markdown("### ⚙️ Session Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset All", help="Clear all data and start fresh"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # App info
    st.markdown("### ℹ️ App Info")
    st.markdown(f"""
    - **Model**: Google Gemini 2.0 Flash
    - **Embeddings**: Text Embedding 004
    - **Vector Store**: FAISS
    - **Version**: 2.0.0
    """)
    
    if check_streamlit_cloud():
        st.success("☁️ Running on Streamlit Cloud")
    else:
        st.info("💻 Running locally")

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💬 Chat with Your Documents")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for j, source in enumerate(message["sources"]):
                            st.write(f"**Source {j+1}:**")
                            st.write(source)
    
    # Chat input
    if st.session_state.conversation_chain:
        user_question = st.chat_input("Ask a question about your documents...")
        
        if user_question:
            # Add user message to history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.conversation_chain({
                        "question": user_question
                    })
                    
                    answer = response.get("answer", "I couldn't generate a response.")
                    sources = response.get("source_documents", [])
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": answer
                    }
                    
                    if sources:
                        assistant_message["sources"] = [doc.page_content[:200] + "..." for doc in sources[:3]]
                    
                    st.session_state.chat_history.append(assistant_message)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
            
            st.rerun()
    else:
        st.info("👆 Please upload and process documents first to start chatting!")

with col2:
    st.markdown("### 📈 Chat Statistics")
    
    if st.session_state.chat_history:
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
        
        st.metric("Total Messages", total_messages)
        st.metric("Your Questions", user_messages)
        st.metric("AI Responses", assistant_messages)
    
    if st.session_state.document_chunks:
        st.metric("Document Chunks", len(st.session_state.document_chunks))
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the important dates mentioned?",
        "Who are the main people discussed?",
        "What conclusions are drawn?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question[:20]}"):
            if st.session_state.conversation_chain:
                # Simulate clicking the question
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 AI Document Assistant | Built with Streamlit & Google Gemini</p>
    <p>Upload your documents and start chatting with AI!</p>
</div>
""", unsafe_allow_html=True)



