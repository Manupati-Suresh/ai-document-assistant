# 🤖 AI Document Assistant

A powerful Streamlit application that allows you to upload PDF documents and chat with them using Google's Gemini AI. Built with LangChain for advanced document processing and retrieval-augmented generation (RAG).

## ✨ Features

- **📁 Multi-file Upload**: Upload multiple PDF documents simultaneously
- **🧠 Smart Chunking**: Intelligent text splitting for optimal processing
- **💬 Conversational AI**: Chat with your documents using natural language
- **🔍 Source Citations**: See which parts of your documents the AI is referencing
- **💾 Persistent Memory**: Conversation history maintained throughout your session
- **📊 Real-time Statistics**: Track your chat activity and document processing
- **🎨 Beautiful UI**: Modern, responsive design with custom styling
- **⚡ Fast Processing**: Optimized batch processing with rate limiting
- **🔒 Secure**: Environment-based API key management

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-document-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Get a Google Gemini API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for the next step

3. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your forked repository
   - Set the main file path to `main.py`

4. **Add your API key to Streamlit Secrets**
   - In your Streamlit Cloud app settings, go to "Secrets"
   - Add the following:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

5. **Deploy and enjoy!** 🎉

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload one or more PDF files
2. **Process Documents**: Click "🚀 Process Documents" to analyze your files
3. **Start Chatting**: Ask questions about your documents in the chat interface
4. **Explore Sources**: Click on source citations to see relevant document excerpts
5. **Use Sample Questions**: Try the suggested questions for inspiration

## 🛠️ Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **AI Model**: Google Gemini 2.0 Flash (latest model)
- **Embeddings**: Google Text Embedding 004
- **Vector Store**: FAISS for efficient similarity search
- **Text Processing**: LangChain with recursive character splitting
- **Memory**: Conversation buffer with sliding window

### Key Components
- **Document Processing**: PDF text extraction with error handling
- **Chunking Strategy**: 1000 characters with 200 character overlap
- **Batch Processing**: Rate-limited API calls to prevent quota issues
- **Caching**: Streamlit caching for models and embeddings
- **Session Management**: Persistent state across user interactions

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Streamlit Configuration
The app includes optimized Streamlit configuration in `.streamlit/config.toml`:
- Custom theme colors
- Performance optimizations
- Security settings

## 📊 Performance Features

- **Intelligent Caching**: Models and embeddings are cached for faster responses
- **Batch Processing**: Documents processed in optimized batches
- **Rate Limiting**: Built-in delays to respect API quotas
- **Memory Management**: Efficient conversation memory with sliding window
- **Error Handling**: Comprehensive error handling with user-friendly messages

## 🎨 UI/UX Features

- **Responsive Design**: Works on desktop and mobile devices
- **Dark/Light Theme**: Follows system preferences
- **Progress Indicators**: Real-time feedback during processing
- **Interactive Elements**: Hover effects and smooth transitions
- **Accessibility**: Screen reader friendly with proper ARIA labels

## 🔒 Security

- **Environment Variables**: API keys stored securely
- **Input Validation**: PDF file type validation
- **Error Handling**: Secure error messages without exposing internals
- **Rate Limiting**: Protection against API abuse

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for the powerful AI tools
- [Google AI](https://ai.google/) for the Gemini models
- [FAISS](https://faiss.ai/) for efficient vector search

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-username/ai-document-assistant/issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**Made with ❤️ and AI** | Deploy on [Streamlit Cloud](https://streamlit.io/cloud) in minutes!