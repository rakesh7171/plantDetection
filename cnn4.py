#!/usr/bin/env python
# coding: utf-8

# In[5]:


# cnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Define CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes for apple diseases
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the CNN model
def load_cnn_model(path='cnn_model4.keras'):
    return tf.keras.models.load_model(path)

# Prepare data generator for training and validation
def prepare_data_generators(train_dir, val_dir, batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='sparse'
    )
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='sparse'
    )
    return train_generator, val_generator

# Train and save the model
def train_cnn_model():
    # Prompt user for dataset paths
    train_dir = '/Users/hritik/myenv/archive (2)/train'
    val_dir = '/Users/hritik/myenv/archive (2)/valid'
    model_path = 'cnn_model4.keras'

    model = create_cnn_model()
    train_generator, val_generator = prepare_data_generators(train_dir, val_dir)
    
    checkpoint = ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=[checkpoint]
    )

# If this script is run directly, train the CNN model
if __name__ == '__main__':
    train_cnn_model()


# In[6]:





# In[ ]:




