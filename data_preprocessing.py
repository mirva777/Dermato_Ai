import os
import numpy as np
import pandas as pd
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR = 'HAM10000_images'
METADATA_FILE_PATH = 'HAM10000_metadata.csv'
metadata = pd.read_csv(METADATA_FILE_PATH)
def append_extension(image_id):
    return f"{image_id}.jpg"
metadata['image_id'] = metadata['image_id'].apply(append_extension)

#image check
print(metadata['image_id'].head())
print(os.listdir(DATA_DIR)[:5])

#load images and check
def load_image(img_id, img_dir=DATA_DIR):
    img_path = os.path.join(img_dir, img_id)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img

#process metadata
metadata['image'] = metadata['image_id'].map(load_image)

#one-hot encoding for labels
labels = pd.get_dummies(metadata['dx'], prefix='label')
metadata = pd.concat([metadata, labels], axis=1)

#Dropping digital dx column if not needed
metadata.drop(columns=['dx'], inplace=True)

#splitting dataset
def split_data(metadata):
    train_val_df, test_df = train_test_split(metadata, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42)
    return train_df, val_df, test_df
train_df, val_df, test_df = split_data(metadata)

#Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col='image_id',
    y_col=labels.columns.tolist(),
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32
)

val_generator = datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_DIR,
    x_col='image_id',
    y_col=labels.columns.tolist(),
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=DATA_DIR,
    x_col='image_id',
    y_col=labels.columns.tolist(),
    target_size=(224, 224),
    class_mode='raw',
    batch_size=32,
    shuffle=False
)

#check
print(f"Train images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Test images: {test_generator.samples}")