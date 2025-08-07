"""
Health Check Page for AI Document Assistant
"""

import streamlit as st
import os
import sys
from datetime import datetime
import platform

st.set_page_config(
    page_title="Health Check",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 System Health Check")

# System Information
st.header("📊 System Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Environment")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Platform:** {platform.platform()}")
    st.write(f"**Streamlit Version:** {st.__version__}")
    st.write(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.subheader("Configuration")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        st.success("✅ GEMINI_API_KEY is configured")
        st.write(f"**Key Preview:** {api_key[:10]}...{api_key[-4:]}")
    else:
        st.error("❌ GEMINI_API_KEY not found")
    
    # Check if running on Streamlit Cloud
    if os.getenv('STREAMLIT_SHARING_MODE'):
        st.success("☁️ Running on Streamlit Cloud")
    else:
        st.info("💻 Running locally")

# Dependencies Check
st.header("📦 Dependencies")

required_packages = [
    'streamlit',
    'langchain',
    'langchain_google_genai',
    'PyPDF2',
    'faiss',
    'python-dotenv'
]

for package in required_packages:
    try:
        if package == 'faiss':
            import faiss
            st.success(f"✅ {package} - Version: {faiss.__version__}")
        elif package == 'streamlit':
            st.success(f"✅ {package} - Version: {st.__version__}")
        elif package == 'PyPDF2':
            import PyPDF2
            st.success(f"✅ {package} - Version: {PyPDF2.__version__}")
        else:
            __import__(package)
            st.success(f"✅ {package} - Installed")
    except ImportError:
        st.error(f"❌ {package} - Not installed")
    except Exception as e:
        st.warning(f"⚠️ {package} - Error: {str(e)}")

# Memory and Performance
st.header("⚡ Performance")

try:
    import psutil
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU Usage", f"{cpu_percent}%")
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric("Memory Usage", f"{memory.percent}%")
    
    with col3:
        disk = psutil.disk_usage('/')
        st.metric("Disk Usage", f"{disk.percent}%")
        
except ImportError:
    st.info("Install psutil for detailed performance metrics")

# Test API Connection
st.header("🔗 API Connection Test")

if st.button("Test Gemini API Connection"):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        with st.spinner("Testing API connection..."):
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                google_api_key=os.getenv('GEMINI_API_KEY'),
                temperature=0.1
            )
            
            # Simple test query
            response = model.invoke("Hello, this is a test. Please respond with 'API connection successful!'")
            
            if "successful" in response.content.lower():
                st.success("✅ API connection test passed!")
                st.write(f"**Response:** {response.content}")
            else:
                st.warning("⚠️ API responded but with unexpected content")
                st.write(f"**Response:** {response.content}")
                
    except Exception as e:
        st.error(f"❌ API connection test failed: {str(e)}")

# File System Check
st.header("📁 File System")

# Check if we can write files
try:
    test_file = "health_check_test.txt"
    with open(test_file, 'w') as f:
        f.write("test")
    os.remove(test_file)
    st.success("✅ File system write access - OK")
except Exception as e:
    st.error(f"❌ File system write access - Failed: {str(e)}")

# Check ChromaDB directory
chroma_dirs = [d for d in os.listdir('.') if d.startswith('chroma_db_')]
if chroma_dirs:
    st.success(f"✅ ChromaDB directories found: {len(chroma_dirs)}")
    for dir_name in chroma_dirs[:3]:  # Show first 3
        st.write(f"**Directory:** {dir_name}")
else:
    st.info("ℹ️ No ChromaDB directories found (normal for first run)")

st.markdown("---")
st.markdown("**Health check completed!** If all items show ✅, your system is ready to go.")