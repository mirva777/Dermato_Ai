import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

from keras._tf_keras.keras.applications import EfficientNetB0
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D
from keras._tf_keras.keras.optimizers import Adam

def create_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax') 
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
