import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Loading the model
model = load_model('model.h5', compile=False)

# Compiling the model
optimizer = Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Seven classes
class_names = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# Process image
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize image to [0, 1]
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    target_size = (224, 224)
    img_array = preprocess_image(file_path, target_size)

    # Predictions
    predictions = model.predict(img_array)
    threshold = 0.5

    # Predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class_confidence = predictions[0][predicted_class_index]

    # Statement to determine the illness
    if predicted_class_confidence > threshold:
        predicted_class_name = class_names[predicted_class_index]
        result = {
            'Predicted Class': predicted_class_name,
            'Confidence': f"{predicted_class_confidence:.2f}"
        }
    else:
        result = {'message': 'No illness detected with sufficient confidence.'}

    # Probability for each class
    probabilities = {class_names[idx]: f"{predictions[0][idx]:.2f}" for idx in class_names}
    result['Probabilities'] = probabilities

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
