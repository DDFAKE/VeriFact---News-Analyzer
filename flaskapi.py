import os
from pathlib import Path
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import joblib
import numpy as np
import re
from textblob import TextBlob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/health": {"origins": "*"}
})

# Load model and vectorizer
try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'fake_news_detector_20241124_221416.joblib')  # Update path as needed
    loaded_data = joblib.load(MODEL_PATH)
    model = loaded_data['model']
    vectorizer = loaded_data['vectorizer']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Basic text preprocessing"""
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    """Simple sentiment analysis"""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check"""
    return jsonify({
        'status': 'healthy' if model and vectorizer else 'error',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input text
        data = request.get_json()
        print(request.get_data())
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Text too short'}), 400

        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Get prediction
        text_vectorized = vectorizer.transform([processed_text])
        prediction = int(model.predict(text_vectorized)[0])
        probabilities = model.predict_proba(text_vectorized)[0].tolist()
        sentiment = analyze_sentiment(text)

        # Prepare response
        response = {
            'prediction': {
                'label': 'Real News' if prediction == 1 else 'Fake News',
                'is_fake': bool(prediction == 0),
                'confidence': float(max(probabilities))
            },
            'probability': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'sentiment': {
                'polarity': float(sentiment['polarity']),
                'subjectivity': float(sentiment['subjectivity'])
            },
            'text_stats': {
                'length': len(text),
                'word_count': len(text.split())
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not model or not vectorizer:
        logger.error("Model not loaded")
        exit(1)

    print("\nStarting Fake News Detection API...")
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port)