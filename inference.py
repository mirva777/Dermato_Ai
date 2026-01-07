"""
Unified Inference Script for Skin Lesion Classification
Supports both TensorFlow/Keras and MindSpore models
"""

import os
import numpy as np
from PIL import Image
import config


class ModelInference:
    """Base class for model inference"""
    
    def __init__(self, model_path, framework='tensorflow'):
        self.model_path = model_path
        self.framework = framework
        self.model = None
        self.class_names = config.CLASS_NAMES
        
    def load_model(self):
        """Load model - to be implemented by subclasses"""
        raise NotImplementedError
    
    def preprocess_image(self, image_path):
        """Preprocess image - to be implemented by subclasses"""
        raise NotImplementedError
    
    def predict(self, image_path):
        """Make prediction - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_top_predictions(self, predictions, top_k=3):
        """
        Get top K predictions
        
        Args:
            predictions: Model predictions (probabilities)
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (class_name, probability)
        """
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = []
        
        for idx in top_indices:
            class_name = self.class_names[idx]
            probability = predictions[idx]
            top_predictions.append((class_name, probability))
        
        return top_predictions


class TensorFlowInference(ModelInference):
    """TensorFlow/Keras model inference"""
    
    def __init__(self, model_path=config.TF_MODEL_PATH):
        super().__init__(model_path, framework='tensorflow')
        self.load_model()
    
    def load_model(self):
        """Load TensorFlow/Keras model"""
        from tensorflow import keras
        from keras.optimizers import Adam
        
        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
            optimizer = Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"TensorFlow model loaded from: {self.model_path}")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for TensorFlow model
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        from keras.preprocessing import image as keras_image
        
        img = keras_image.load_img(image_path, target_size=config.IMAGE_SIZE)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        return img_array
    
    def predict(self, image_path, return_top_k=3):
        """
        Make prediction using TensorFlow model
        
        Args:
            image_path: Path to image file
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top predictions
        top_predictions = self.get_top_predictions(predictions, top_k=return_top_k)
        
        # Prepare results
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        result = {
            'framework': 'TensorFlow',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_predictions': [
                {'class': cls, 'probability': float(prob)}
                for cls, prob in top_predictions
            ],
            'all_probabilities': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(predictions))
            }
        }
        
        return result


class MindSporeInference(ModelInference):
    """MindSpore model inference"""
    
    def __init__(self, model_path=config.MINDSPORE_MODEL_PATH, model_type='resnet'):
        super().__init__(model_path, framework='mindspore')
        self.model_type = model_type
        self.load_model()
    
    def load_model(self):
        """Load MindSpore model"""
        import mindspore
        from mindspore import context, load_checkpoint, load_param_into_net
        from model_mindspore import create_model
        
        try:
            # Set context
            context.set_context(
                mode=context.GRAPH_MODE,
                device_target=config.MINDSPORE_DEVICE_TARGET
            )
            
            # Create model architecture
            self.model = create_model(
                model_type=self.model_type,
                num_classes=config.NUM_CLASSES
            )
            
            # Load weights
            if os.path.exists(self.model_path):
                param_dict = load_checkpoint(self.model_path)
                load_param_into_net(self.model, param_dict)
                print(f"MindSpore model loaded from: {self.model_path}")
            else:
                print(f"Warning: Model file not found at {self.model_path}")
                print("Using untrained model!")
            
            # Set to eval mode
            self.model.set_train(False)
            
        except Exception as e:
            print(f"Error loading MindSpore model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for MindSpore model
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        import mindspore
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        img = img.resize(config.IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406]) * 255
        std = np.array([0.229, 0.224, 0.225]) * 255
        img_array = (img_array - mean) / std
        
        # Convert to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return mindspore.Tensor(img_array, mindspore.float32)
    
    def predict(self, image_path, return_top_k=3):
        """
        Make prediction using MindSpore model
        
        Args:
            image_path: Path to image file
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        import mindspore.nn as nn
        
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        logits = self.model(img_tensor)
        
        # Apply softmax
        softmax = nn.Softmax(axis=1)
        predictions = softmax(logits).asnumpy()[0]
        
        # Get top predictions
        top_predictions = self.get_top_predictions(predictions, top_k=return_top_k)
        
        # Prepare results
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        result = {
            'framework': 'MindSpore',
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'top_predictions': [
                {'class': cls, 'probability': float(prob)}
                for cls, prob in top_predictions
            ],
            'all_probabilities': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(predictions))
            }
        }
        
        return result


class EnsembleInference:
    """Ensemble inference using both TensorFlow and MindSpore models"""
    
    def __init__(self, tf_model_path=config.TF_MODEL_PATH, 
                 ms_model_path=config.MINDSPORE_MODEL_PATH,
                 ms_model_type='resnet'):
        self.tf_inference = None
        self.ms_inference = None
        
        # Try to load TensorFlow model
        try:
            self.tf_inference = TensorFlowInference(tf_model_path)
        except Exception as e:
            print(f"Could not load TensorFlow model: {e}")
        
        # Try to load MindSpore model
        try:
            self.ms_inference = MindSporeInference(ms_model_path, ms_model_type)
        except Exception as e:
            print(f"Could not load MindSpore model: {e}")
        
        if self.tf_inference is None and self.ms_inference is None:
            raise RuntimeError("Failed to load any model!")
    
    def predict(self, image_path, weights=(0.5, 0.5), return_top_k=3):
        """
        Make ensemble prediction
        
        Args:
            image_path: Path to image file
            weights: Weights for (TensorFlow, MindSpore) models
            return_top_k: Number of top predictions to return
            
        Returns:
            Dictionary with ensemble prediction results
        """
        predictions = []
        frameworks = []
        
        # Get TensorFlow predictions
        if self.tf_inference is not None:
            tf_result = self.tf_inference.predict(image_path, return_top_k=config.NUM_CLASSES)
            tf_probs = np.array([tf_result['all_probabilities'][config.CLASS_NAMES[i]] 
                                for i in range(config.NUM_CLASSES)])
            predictions.append(tf_probs * weights[0])
            frameworks.append('TensorFlow')
        
        # Get MindSpore predictions
        if self.ms_inference is not None:
            ms_result = self.ms_inference.predict(image_path, return_top_k=config.NUM_CLASSES)
            ms_probs = np.array([ms_result['all_probabilities'][config.CLASS_NAMES[i]] 
                                for i in range(config.NUM_CLASSES)])
            predictions.append(ms_probs * weights[1])
            frameworks.append('MindSpore')
        
        # Ensemble predictions
        ensemble_probs = np.sum(predictions, axis=0)
        ensemble_probs /= np.sum(ensemble_probs)  # Normalize
        
        # Get top predictions
        top_indices = np.argsort(ensemble_probs)[::-1][:return_top_k]
        top_predictions = [
            {'class': config.CLASS_NAMES[idx], 'probability': float(ensemble_probs[idx])}
            for idx in top_indices
        ]
        
        predicted_class_idx = np.argmax(ensemble_probs)
        
        result = {
            'framework': f'Ensemble ({", ".join(frameworks)})',
            'predicted_class': config.CLASS_NAMES[predicted_class_idx],
            'confidence': float(ensemble_probs[predicted_class_idx]),
            'top_predictions': top_predictions,
            'all_probabilities': {
                config.CLASS_NAMES[i]: float(ensemble_probs[i])
                for i in range(config.NUM_CLASSES)
            }
        }
        
        return result


def predict_single_image(image_path, framework='auto', model_type='resnet'):
    """
    Convenient function to make prediction on a single image
    
    Args:
        image_path: Path to image file
        framework: 'tensorflow', 'mindspore', 'ensemble', or 'auto'
        model_type: Model type for MindSpore ('resnet' or 'efficientnet')
        
    Returns:
        Dictionary with prediction results
    """
    if framework == 'tensorflow':
        inference = TensorFlowInference()
        return inference.predict(image_path)
    
    elif framework == 'mindspore':
        inference = MindSporeInference(model_type=model_type)
        return inference.predict(image_path)
    
    elif framework == 'ensemble':
        inference = EnsembleInference(ms_model_type=model_type)
        return inference.predict(image_path)
    
    else:  # auto
        # Try to use ensemble, fall back to available model
        try:
            inference = EnsembleInference(ms_model_type=model_type)
            return inference.predict(image_path)
        except:
            try:
                inference = TensorFlowInference()
                return inference.predict(image_path)
            except:
                inference = MindSporeInference(model_type=model_type)
                return inference.predict(image_path)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Skin lesion inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--framework', type=str, default='auto',
                       choices=['tensorflow', 'mindspore', 'ensemble', 'auto'],
                       help='Framework to use for inference')
    parser.add_argument('--model_type', type=str, default='resnet',
                       choices=['resnet', 'efficientnet'],
                       help='Model type for MindSpore')
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_single_image(
        args.image,
        framework=args.framework,
        model_type=args.model_type
    )
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Prediction Results - Framework: {result['framework']}")
    print("=" * 80)
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    print(f"\nTop {len(result['top_predictions'])} Predictions:")
    for i, pred in enumerate(result['top_predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['probability']:.2%}")
    
    print("\n" + "=" * 80)
