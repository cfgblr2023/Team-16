from flask import Flask, jsonify, request
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

data_path = "C:\cfg\Photos_Walking"
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10)
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    subset='training')

# Load the saved model
saved_model = load_model('model.h5')

# Get the class labels from the generator
class_labels = list(train_generator.class_indices.keys())

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    file = request.files['image']
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    
    # Predict the class probabilities for the input image
    predictions = saved_model.predict(img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]
    
    # Return the predicted class as a response
    response = {'predicted_class': predicted_class}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
