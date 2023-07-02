import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np

# Set the path to your dataset
data_path = "C:\cfg\Photos_Walking"

# Set the number of classes in your dataset
num_classes = 17

# Set the input image size
input_shape = (224, 224, 3)

# Set the batch size and number of epochs
batch_size = 32
epochs = 20

# Load the ResNet-50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Add your own layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Add more layers and normalization
x = Dense(256, activation='relu')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)

predictions = Dense(num_classes, activation='softmax')(x)

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=10)

# Load and preprocess the data
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    subset='training')

valid_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    subset='validation')

# Fine-tune the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size)

# Save the model as a .h5 file
model.save('model.h5')

# Plot validation accuracy
plt.plot(history.history['val_accuracy'])
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Load the saved model
saved_model = load_model('model.h5')

# Load and preprocess the input image
input_image = tf.keras.preprocessing.image.load_img('input.jpg', target_size=(224, 224))
input_image = tf.keras.preprocessing.image.img_to_array(input_image)
input_image = np.expand_dims(input_image, axis=0)
input_image = input_image / 255.0

# Predict the class probabilities for the input image
predictions = saved_model.predict(input_image)
predicted_class = np.argmax(predictions[0])

# Print the predicted class
print("Predicted Class:",  list(train_generator.class_indices.keys())[predicted_class])
