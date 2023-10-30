import tensorflow as tf
import numpy as np
#from keras.utils import image_utils
from keras.preprocessing.image import ImageDataGenerator


#train_ds = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
train_ds = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_ds = train_ds.flow_from_directory('C:\Users\Ben\Desktop\Data Science\Program\Image Classification\training_set/training_set',target_size=(64, 64),batch_size=32,class_mode='binary')

#test_ds = ImageDataGenerator(rescale = 1./255)
test_ds = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_ds = test_ds.flow_from_directory('C:\Users\Ben\Desktop\Data Science\Program\Image Classification\test_set/test_set/',target_size=(64, 64),batch_size=32,class_mode='binary')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

model.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=128, activation='relu'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_ds, validation_data=test_ds, epochs=25)

image_utils.load_img('C:\Users\Ben\Desktop\Data Science\Program\Image Classification\test_set/test_set/dogs/dog.4462.jpg')

test_img = image_utils.load_img('C:\Users\Ben\Desktop\Data Science\Program\Image Classification\test_set/test_set/dogs/dog.4462.jpg', target_size = (64, 64))
img = image_utils.img_to_array(test_img)
img = np.expand_dims(img, axis = 0)
r = model.predict(img)
train_ds.class_indices
if r[0][0] == 1:
    pred = 'dog'
else:
    pred = 'cat'

print(pred)