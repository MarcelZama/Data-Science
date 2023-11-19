# Import necessary libraries
from keras.src.preprocessing.image import image_utils
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Define an ImageDataGenerator for data augmentation during training
train_ds = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True
)
# Create a generator for the training dataset
train_ds = train_ds.flow_from_directory(
    '/Users/marcis578/Documents/University/Data_Science/OneDrive_2023-11-10/Image Classification/training_set/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Define an ImageDataGenerator for normalizing pixel values during testing
test_ds = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# Create a generator for the testing dataset
test_ds = test_ds.flow_from_directory(
    '/Users/marcis578/Documents/University/Data_Science/OneDrive_2023-11-10/Image Classification/test_set/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Build a sequential model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training dataset and validate on the testing dataset
model.fit(x=train_ds, validation_data=test_ds, epochs=25)

# Load and preprocess a single test image for prediction
test_img = image_utils.load_img('/Users/marcis578/Documents/University/Data_Science/OneDrive_2023-11-10/Image Classification/test_set/test_set/dogs/dog.4462.jpg', target_size=(64, 64))
img = image_utils.img_to_array(test_img)
img = np.expand_dims(img, axis=0)

# Make a prediction using the trained model
r = model.predict(img)

# Get the class indices from the training dataset
class_indices = train_ds.class_indices

# Determine the predicted class (dog or cat) based on the model's output
if r[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'

# Print the predicted class
print(pred)
