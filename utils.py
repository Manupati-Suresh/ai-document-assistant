"""
Utility functions for the AI Document Assistant
"""

import os
import streamlit as st
import logging
from typing import Optional, Dict, Any
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_api_key() -> bool:
    """
    Check if the Gemini API key is available and valid
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables!")
        st.info("Please add your Google Gemini API key to continue.")
        return False
    
    # Basic validation - check if it looks like a valid API key
    if not api_key.startswith('AIza') or len(api_key) < 30:
        st.error("⚠️ Invalid API key format. Please check your GEMINI_API_KEY.")
        return False
    
    return True

def validate_pdf_file(file) -> bool:
    """
    Validate uploaded PDF file
    """
    if not file:
        return False
    
    # Check file type
    if not file.type == "application/pdf":
        st.error("❌ Please upload a PDF file only.")
        return False
    
    # Check file size (limit to 50MB)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    if file.size > max_size:
        st.error(f"❌ File too large. Maximum size is 50MB. Your file is {file.size / (1024*1024):.1f}MB.")
        return False
    
    return True

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "streamlit_version": st.__version__,
        "python_version": os.sys.version,
        "has_api_key": bool(os.getenv('GEMINI_API_KEY')),
    }

def display_error_info(error: Exception, context: str = ""):
    """
    Display user-friendly error information
    """
    error_msg = str(error)
    
    # Common error patterns and user-friendly messages
    if "429" in error_msg or "quota" in error_msg.lower():
        st.error("⚠️ API quota exceeded. Please wait a moment and try again.")
        st.info("💡 Tip: Try processing smaller documents or wait a few minutes before retrying.")
    
    elif "401" in error_msg or "unauthorized" in error_msg.lower():
        st.error("🔑 API key authentication failed. Please check your GEMINI_API_KEY.")
        st.info("💡 Tip: Make sure your API key is valid and has the necessary permissions.")
    
    elif "connection" in error_msg.lower() or "network" in error_msg.lower():
        st.error("🌐 Network connection error. Please check your internet connection.")
        st.info("💡 Tip: Try refreshing the page or check your network connection.")
    
    elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
        st.error("💾 Memory limit exceeded. Try processing smaller documents.")
        st.info("💡 Tip: Break large documents into smaller files or reduce the number of files processed at once.")
    
    else:
        st.error(f"❌ An error occurred: {error_msg}")
        if context:
            st.info(f"Context: {context}")
    
    # Log the full error for debugging
    logger.error(f"Error in {context}: {error}", exc_info=True)

def create_download_link(data: str, filename: str, link_text: str) -> str:
    """
    Create a download link for text data
    """
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    """
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext
    
    return filename

class ProgressTracker:
    """
    Simple progress tracking utility
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step: int, message: str = ""):
        self.current_step = step
        progress = min(step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        if message:
            self.status_text.text(f"{self.description}: {message}")
        else:
            self.status_text.text(f"{self.description}: Step {step}/{self.total_steps}")
    
    def complete(self, message: str = "Complete!"):
        self.progress_bar.progress(1.0)
        self.status_text.text(message)
    
    def cleanup(self):
        self.progress_bar.empty()
        self.status_text.empty()

def check_streamlit_cloud() -> bool:
    """
    Check if running on Streamlit Cloud
    """
    return os.getenv('STREAMLIT_SHARING_MODE') is not None

def get_app_url() -> Optional[str]:
    """
    Get the current app URL if available
    """
    try:
        # This works in Streamlit Cloud
        return st.experimental_get_query_params().get('url', [None])[0]
    except:
        return None