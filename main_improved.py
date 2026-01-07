"""
Improved Flask API for Skin Lesion Classification
Features: Better error handling, logging, support for multiple frameworks
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests 
import json

# Import configuration and inference module
import config
from inference import predict_single_image, TensorFlowInference, MindSporeInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'api.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize model (lazy loading)
_model_cache = {}


def get_model(framework='tensorflow', model_type='resnet'):
    """
    Get or create model instance (cached)
    
    Args:
        framework: 'tensorflow' or 'mindspore'
        model_type: Model architecture type
        
    Returns:
        Model instance
    """
    cache_key = f"{framework}_{model_type}"
    
    if cache_key not in _model_cache:
        try:
            if framework == 'tensorflow':
                _model_cache[cache_key] = TensorFlowInference()
                logger.info("TensorFlow model loaded successfully")
            elif framework == 'mindspore':
                _model_cache[cache_key] = MindSporeInference(model_type=model_type)
                logger.info(f"MindSpore {model_type} model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {framework} model: {e}")
            raise
    
    return _model_cache[cache_key]


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_medical_advice(prediction_result):
    """
    Get medical advice from external AI service
    
    Args:
        prediction_result: Dictionary with prediction results
        
    Returns:
        String with medical advice or None if service unavailable
    """
    try:
        prompt = {
            "request": json.dumps(prediction_result) + 
            " if you get this type of prompt, this is response from our skin scanner AI. " +
            "Please provide: 1) A brief explanation of the predicted condition, " +
            "2) The confidence level interpretation, " +
            "3) Recommended type of doctor to consult (e.g., dermatologist), " +
            "4) General care recommendations, " +
            "5) Important disclaimer that this AI is for screening purposes only and " +
            "cannot replace professional medical diagnosis. Maintain a professional, " +
            "reassuring tone."
        }
        
        external_url = 'https://web.binaryhood.uz/api/v1/chat/generate'
        response = requests.post(external_url, json=prompt, timeout=10)
        response.raise_for_status()
        
        outer_dict = json.loads(response.content.decode('utf-8'))
        return outer_dict.get("response", None)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with AI service: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_medical_advice: {e}")
        return None


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Dermato AI - Skin Lesion Classifier'
    }), 200


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    available_models = []
    
    # Check TensorFlow model
    if os.path.exists(config.TF_MODEL_PATH):
        available_models.append({
            'framework': 'tensorflow',
            'path': config.TF_MODEL_PATH,
            'type': 'EfficientNetB0'
        })
    
    # Check MindSpore model
    if os.path.exists(config.MINDSPORE_MODEL_PATH):
        available_models.append({
            'framework': 'mindspore',
            'path': config.MINDSPORE_MODEL_PATH,
            'type': 'ResNet-CBAM'
        })
    
    return jsonify({
        'available_models': available_models,
        'num_classes': config.NUM_CLASSES,
        'class_names': config.CLASS_NAMES
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Expected form data:
        - file: Image file
        - framework: (optional) 'tensorflow', 'mindspore', or 'auto' (default: 'auto')
        - model_type: (optional) 'resnet' or 'efficientnet' (default: 'resnet')
        - get_advice: (optional) 'true' or 'false' (default: 'true')
    
    Returns:
        JSON with prediction results and optional medical advice
    """
    start_time = datetime.now()
    
    # Validate request
    if 'file' not in request.files:
        logger.warning("Prediction request missing file")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("Prediction request with empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {filename}")
        
        # Get parameters
        framework = request.form.get('framework', 'auto')
        model_type = request.form.get('model_type', 'resnet')
        get_advice = request.form.get('get_advice', 'true').lower() == 'true'
        
        # Make prediction
        logger.info(f"Making prediction with framework={framework}, model_type={model_type}")
        prediction_result = predict_single_image(
            file_path,
            framework=framework,
            model_type=model_type
        )
        
        # Check confidence threshold
        confidence = prediction_result['confidence']
        predicted_class = prediction_result['predicted_class']
        
        if confidence < config.CONFIDENCE_THRESHOLD:
            logger.info(f"Low confidence prediction: {confidence:.2%}")
            result = {
                'status': 'low_confidence',
                'message': 'No illness detected with sufficient confidence.',
                'confidence': confidence,
                'predicted_class': predicted_class,
                'top_predictions': prediction_result['top_predictions'],
                'warning': 'Please consult a dermatologist for professional diagnosis.'
            }
        else:
            result = {
                'status': 'success',
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top_predictions': prediction_result['top_predictions'],
                'framework': prediction_result['framework']
            }
        
        # Get medical advice if requested
        if get_advice:
            logger.info("Requesting medical advice from external service")
            advice = get_medical_advice(prediction_result)
            
            if advice:
                result['medical_advice'] = advice
                result['title'] = predicted_class
                result['response'] = advice
            else:
                result['medical_advice'] = (
                    f"Predicted condition: {predicted_class}. "
                    f"Confidence: {confidence:.1%}. "
                    "Please consult a dermatologist for professional diagnosis. "
                    "This AI tool is for screening purposes only and should not "
                    "replace professional medical advice."
                )
        
        # Add metadata
        result['metadata'] = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
        
        logger.info(f"Prediction successful: {predicted_class} ({confidence:.2%})")
        
        # Clean up uploaded file (optional)
        # os.remove(file_path)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error during prediction',
            'details': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected form data:
        - files[]: Multiple image files
        - framework: (optional) 'tensorflow', 'mindspore', or 'auto'
    
    Returns:
        JSON with prediction results for all images
    """
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    framework = request.form.get('framework', 'auto')
    model_type = request.form.get('model_type', 'resnet')
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Predict
                prediction = predict_single_image(
                    file_path,
                    framework=framework,
                    model_type=model_type
                )
                
                results.append({
                    'filename': file.filename,
                    'prediction': prediction,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'status': 'error'
                })
        else:
            results.append({
                'filename': file.filename if file else 'unknown',
                'error': 'Invalid file type',
                'status': 'error'
            })
    
    return jsonify({
        'results': results,
        'total': len(files),
        'successful': sum(1 for r in results if r['status'] == 'success')
    }), 200


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Dermato AI Server...")
    logger.info(f"Upload folder: {config.UPLOAD_FOLDER}")
    logger.info(f"Log directory: {config.LOG_DIR}")
    
    # Pre-load model
    try:
        get_model(framework='tensorflow')
        logger.info("Model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
        logger.warning("Model will be loaded on first request")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
