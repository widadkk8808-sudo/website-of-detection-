"""
Improved Phishing Detection Flask API
AI-Driven Awareness Program for Phishing Email Detection using SVM
Author: MiniMax Agent
Python Version: 3.13.4
Minimum Supported: Python 3.7.3
Created: 2025-11-24
Enhanced Version with improved model accuracy
"""


from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import sys
import os

# Import our improved phishing detector
from improved_phishing_detector import detector

# Version Information
PYTHON_VERSION = "3.13.4"
MIN_PYTHON_VERSION = "3.7.3"
API_VERSION = "1.2"
MODEL_VERSION = "improved_secure"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
def load_model():
    try:
        with open('svm_phishing_model.pkl', 'rb') as f:
            detector.pipeline = pickle.load(f)
        detector.is_trained = True
        print("Improved model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading improved model: {e}")
        return False

# Load model on startup
model_loaded = load_model()

@app.route('/')
def serve_index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('.', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'message': 'Improved Phishing Detection API is running',
        'version_info': {
            'api_version': API_VERSION,
            'model_version': MODEL_VERSION,
            'python_version': PYTHON_VERSION,
            'minimum_python': MIN_PYTHON_VERSION
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_email():
    """Predict if an email is phishing or safe using improved model with security fixes"""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        # Get email text from request
        data = request.get_json()
        if not data or 'email_text' not in data:
            return jsonify({
                'error': 'Missing email_text in request',
                'status': 'error'
            }), 400
        
        email_text = data['email_text']
        if not email_text or email_text.strip() == '':
            return jsonify({
                'error': 'Email text cannot be empty',
                'status': 'error'
            }), 400
        
        # Make prediction using improved model with security fixes
        result = detector.predict_email(email_text)
        
        # Handle security validation errors
        if result['prediction'] == 'Error':
            return jsonify({
                'status': 'error',
                'error': result.get('error_message', 'Input validation failed'),
                'security_validation': result.get('security_validation', 'failed'),
                'model_version': MODEL_VERSION,
                'prediction': 'Error',
                'confidence': 0
            }), 400
        
        # Generate explanation for normal cases
        explanation = detector.generate_detailed_explanation(email_text)
        
        return jsonify({
            'status': 'success',
            'model_version': MODEL_VERSION,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'safe_probability': result['safe_probability'],
            'phishing_probability': result['phishing_probability'],
            'security_validation': result.get('security_validation', 'unknown'),
            'context_analysis': result.get('context_analysis'),
            'explanation': explanation,
            'raw_result': result
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Get detailed explanation for email classification using improved model"""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        # Get email text from request
        data = request.get_json()
        if not data or 'email_text' not in data:
            return jsonify({
                'error': 'Missing email_text in request',
                'status': 'error'
            }), 400
        
        email_text = data['email_text']
        if not email_text or email_text.strip() == '':
            return jsonify({
                'error': 'Email text cannot be empty',
                'status': 'error'
            }), 400
        
        # Generate explanation using improved model
        explanation = detector.generate_detailed_explanation(email_text)
        
        return jsonify({
            'status': 'success',
            'model_version': MODEL_VERSION,
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the improved trained model"""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        return jsonify({
            'status': 'success',
            'model_type': 'Support Vector Machine (SVM) - Improved Version',
            'algorithm': 'Linear SVM with Enhanced TF-IDF Vectorization',
            'training_method': 'Enhanced dataset with 40 phishing + 20 safe emails',
            'features': 'Enhanced TF-IDF vectorization with bigrams and improved parameters',
            'accuracy_improvements': [
                'Increased training data (40 vs 13 phishing samples)',
                'Enhanced TF-IDF parameters (max_features=500, ngram_range=(1,2))',
                'Improved SVM configuration (C=2.0)',
                'Better text preprocessing and feature extraction',
                'Confidence boosting for obvious phishing patterns'
            ],
            'description': 'AI-Driven Awareness Program for Phishing Email Detection using SVM (Improved Version)',
            'version_info': {
                'python_version': PYTHON_VERSION,
                'minimum_python': MIN_PYTHON_VERSION,
                'api_version': API_VERSION,
                'model_version': MODEL_VERSION
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    print("=== Improved Phishing Detection API ===")
    print(f"Python Version: {PYTHON_VERSION}")
    print(f"Minimum Supported: {MIN_PYTHON_VERSION}")
    print(f"API Version: {API_VERSION}")
    print(f"Model Version: {MODEL_VERSION}")
    print("="*45)
    print("Starting Improved Phishing Detection API...")
    print(f"Model loaded: {model_loaded}")
    
    if model_loaded:
        print("Improved API is ready to accept requests!")
        print("Endpoints:")
        print("  GET  /api/health - Health check")
        print("  GET  /api/model-info - Model information")
        print("  POST /api/predict - Predict email classification (improved)")
        print("  POST /api/explain - Get detailed explanation (improved)")
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
