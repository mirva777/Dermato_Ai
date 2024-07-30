import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam

#Loading model
model = load_model('model.h5', compile=False)
#Compiling model
optimizer = Adam(learning_rate=0.005)  
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#seven classes
class_names = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'

}

#Process image
def preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  #Normalize image to [0, 1]
    return img_array
image_path = '/Users/mirvohid/Downloads/ezgif-3-4ff7be8053.jpg'
target_size = (224, 224)
img_array = preprocess_image(image_path, target_size)

#Predictions
predictions = model.predict(img_array)
print("Predictions:", predictions)
threshold = 0.5

#Predicted class 
predicted_class_index = np.argmax(predictions)
predicted_class_confidence = predictions[0][predicted_class_index] #its index

#Statement to determine the ilness
if predicted_class_confidence > threshold:
    predicted_class_name = class_names[predicted_class_index]
    print(f"Predicted Class: {predicted_class_name} with confidence {predicted_class_confidence:.2f}")
else:
    print("No illness detected with sufficient confidence.")

#probability for each class
for idx, class_name in enumerate(class_names):
    print(f"{class_name}: {predictions[0][idx]:.2f}")
