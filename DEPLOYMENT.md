# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your AI Document Assistant to Streamlit Cloud in just a few minutes.

## 📋 Prerequisites

1. **GitHub Account**: You'll need a GitHub account to host your code
2. **Google Gemini API Key**: Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## 🔧 Step-by-Step Deployment

### Step 1: Prepare Your Repository

1. **Fork or Clone** this repository to your GitHub account
2. **Ensure all files are present**:
   - `main.py` (main application)
   - `requirements.txt` (dependencies)
   - `.streamlit/config.toml` (Streamlit configuration)
   - `README.md` (documentation)

### Step 2: Get Your Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key (keep it secure!)

### Step 3: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in with GitHub"

2. **Create New App**
   - Click "New app"
   - Select "From existing repo"
   - Choose your GitHub repository
   - Set the main file path to `main.py`
   - Choose a custom URL (optional)

3. **Configure Secrets**
   - Before deploying, click "Advanced settings"
   - Go to the "Secrets" section
   - Add your API key:
     ```toml
     GEMINI_API_KEY = "your_actual_api_key_here"
     ```

4. **Deploy**
   - Click "Deploy!"
   - Wait for the deployment to complete (usually 2-3 minutes)

### Step 4: Verify Deployment

1. **Test the Application**
   - Upload a sample PDF
   - Process the document
   - Ask a few questions
   - Verify responses are working

2. **Check Logs** (if needed)
   - Click on "Manage app" in Streamlit Cloud
   - View logs for any errors

## 🔧 Configuration Options

### Environment Variables (Secrets)

Add these to your Streamlit Cloud secrets:

```toml
# Required
GEMINI_API_KEY = "your_google_gemini_api_key"

# Optional - for advanced configurations
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
MAX_BATCH_SIZE = "10"
```

### Streamlit Configuration

The app includes optimized settings in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

## 🚨 Troubleshooting

### Common Issues and Solutions

1. **"Module not found" errors**
   - Check that all dependencies are in `requirements.txt`
   - Ensure version compatibility

2. **API Key errors**
   - Verify the API key is correctly set in Streamlit secrets
   - Check that the key has proper permissions

3. **Memory issues**
   - Large PDFs might cause memory problems
   - Consider implementing file size limits

4. **Slow processing**
   - This is normal for large documents
   - The app includes progress indicators

### Performance Optimization

1. **Resource Limits**
   - Streamlit Cloud has memory limits
   - Large documents (>10MB) might cause issues
   - Consider implementing file size validation

2. **API Rate Limits**
   - The app includes built-in rate limiting
   - Adjust batch sizes if needed

3. **Caching**
   - Models and embeddings are cached
   - Vector stores are saved locally

## 📊 Monitoring and Maintenance

### Usage Analytics

Streamlit Cloud provides:
- App usage statistics
- Error logs
- Performance metrics

### Updates and Maintenance

1. **Code Updates**
   - Push changes to your GitHub repo
   - Streamlit Cloud auto-deploys on push

2. **Dependency Updates**
   - Update `requirements.txt` as needed
   - Test locally before deploying

3. **API Key Rotation**
   - Update secrets in Streamlit Cloud
   - No code changes needed

## 🔒 Security Best Practices

1. **API Key Management**
   - Never commit API keys to code
   - Use Streamlit secrets for sensitive data
   - Rotate keys regularly

2. **Input Validation**
   - The app validates PDF file types
   - Consider adding file size limits

3. **Error Handling**
   - Errors are handled gracefully
   - No sensitive information exposed

## 📈 Scaling Considerations

### For High Traffic

1. **Consider Streamlit Cloud Pro**
   - Higher resource limits
   - Better performance
   - Priority support

2. **Alternative Deployments**
   - Docker containers
   - Cloud platforms (AWS, GCP, Azure)
   - Kubernetes clusters

### For Enterprise Use

1. **Self-hosted Options**
   - Deploy on your infrastructure
   - Better control over data
   - Custom security policies

2. **Database Integration**
   - Store conversation history
   - User management
   - Analytics and reporting

## 🎉 Success!

Once deployed, your AI Document Assistant will be available at:
`https://your-app-name.streamlit.app`

Share the link with others and start chatting with documents using AI!

## 📞 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Review this troubleshooting guide
3. Open an issue on GitHub
4. Contact Streamlit support for platform issues

---

**Happy Deploying!** 🚀