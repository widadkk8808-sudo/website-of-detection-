# Render Deployment Guide

## Quick Deployment with Python 3.11

### Step 1: Choose Python Version
- **Recommended**: Python 3.11.5 (stable, optimal performance)
- **Alternative**: Python 3.9.x or 3.10.x
- **Avoid**: Python 3.12+ (may have compatibility issues)

### Step 2: Render Configuration

#### Option A: Using render.yaml
```yaml
services:
  - type: web
    name: phishing-detector
    env: python
    pythonVersion: "3.11.5"
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python api.py
    envVars:
      - key: PYTHON_VERSION
        value: "3.11.5"
```

#### Option B: Using Web Dashboard
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +" → "Web Service"
3. Connect your repository
4. **Runtime**: Select "Python 3"
5. **Python Version**: Set to "3.11.5"
6. **Build Command**: `pip install -r requirements.txt`
7. **Start Command**: `python api.py`

#### Option C: Environment Variables (Alternative)
In Render dashboard, set:
```
PYTHON_VERSION=3.11.5
```

### Step 3: Files to Include

Ensure these files are in your repository:

```
📁 phishing-detection-project/
├── 📄 api.py                    (Main API)
├── 📄 working_phishing_detector.py
├── 📄 improved_phishing_detector.py
├── 📄 svm_phishing_model.pkl    (Trained model)
├── 📄 index.html               (Frontend)
├── 📄 script.js
├── 📄 styles.css
├── 📄 requirements.txt         (Dependencies)
└── 📄 render.yaml              (Optional: Render config)
```

### Step 4: Important Configuration

#### requirements.txt (Make sure it includes version info)
```
# Phishing Detection System Requirements
# Python Version: 3.11.5 (Recommended for Render)
# Minimum Supported: Python 3.7.3

flask>=2.3.0,<4.0.0
flask-cors>=4.0.0,<5.0.0
scikit-learn>=1.3.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.24.0,<2.0.0
```

#### API Configuration for Production
Update `api.py` for production:

```python
if __name__ == '__main__':
    # For Render/production
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
```

### Step 5: Testing Before Deployment

1. **Local Test**: Run `python api.py`
2. **Check Version**: Run `python check_python_version.py`
3. **Test API Endpoints**:
   - `GET /api/health` - Should return version info
   - `POST /api/predict` - Should classify emails

### Step 6: Deploy and Verify

1. **Deploy** through Render dashboard
2. **Wait** for build to complete
3. **Visit** your Render URL
4. **Test** the web interface
5. **Check** `/api/health` endpoint

### Step 7: Using the Improved Model

For better accuracy, upload these files to Render:

1. `improved_phishing_detector.py`
2. `api_improved.py`
3. Set start command to: `python api_improved.py`

### Common Issues and Solutions

#### Issue 1: Port Error
**Problem**: `Address already in use`
**Solution**: Update `api.py` to use environment port:
```python
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
```

#### Issue 2: Model Not Found
**Problem**: `svm_phishing_model.pkl not found`
**Solution**: Ensure model file is included in repository

#### Issue 3: Package Compatibility
**Problem**: Package installation fails
**Solution**: Update `requirements.txt` with compatible versions

#### Issue 4: Python Version Mismatch
**Problem**: Version compatibility errors
**Solution**: 
1. Set Python version to 3.11.5 in Render
2. Use compatible package versions in requirements.txt

### Performance Optimization

#### For Better Performance:
1. **Use Python 3.11.5** (optimal for Render)
2. **Enable gzip compression** in Render settings
3. **Set auto-scaling** if needed
4. **Monitor usage** in Render dashboard

### Security Considerations

1. **No sensitive data** in repository
2. **Use environment variables** for any secrets
3. **Keep dependencies updated**
4. **Monitor access logs** in Render

### Cost Optimization

1. **Free Tier**: Suitable for development/testing
2. **Starter Plan**: $7/month for production
3. **Auto-sleep**: Free tier sleeps after 15 min inactivity

### Monitoring and Logs

1. **View logs** in Render dashboard
2. **Monitor metrics** (CPU, memory usage)
3. **Set up alerts** for downtime
4. **Check health endpoint** regularly

### Quick Commands

```bash
# Local testing
python check_python_version.py
python working_phishing_detector.py
python api.py

# Build and deploy
git add .
git commit -m "Deploy phishing detector with Python 3.11.5"
git push origin main
```

### Success Checklist

- [ ] Python version set to 3.11.5 in Render
- [ ] All required files in repository
- [ ] requirements.txt configured correctly
- [ ] API endpoints responding
- [ ] Model loading successfully
- [ ] Web interface accessible
- [ ] Health check passing
- [ ] Version information visible in API response

### Expected API Response

After successful deployment, `/api/health` should return:

```json
{
    "status": "healthy",
    "model_loaded": true,
    "message": "Phishing Detection API is running",
    "version_info": {
        "api_version": "1.0",
        "python_version": "3.11.5",
        "minimum_python": "3.7.3"
    }
}
```

### Support

If you encounter issues:
1. Check Render build logs
2. Verify Python version settings
3. Test locally first
4. Check package compatibility
5. Ensure all files are included

---

## Summary

- **Best Python Version for Render**: 3.11.5
- **Key Configuration**: Set Python version in dashboard
- **Critical Files**: api.py, model file, requirements.txt
- **Testing**: Always test locally before deploying
- **Version Info**: All code now includes version tracking