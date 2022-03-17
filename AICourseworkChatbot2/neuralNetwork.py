# CNN
import kwargs as kwargs
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.preprocessing.image_dataset as id

training_data = id.image_dataset_from_directory(
    directory="C:/Users/ryder/Downloads/Supernatural_Train_Dataset/",
    labels='inferred',
    label_mode='categorical',
    batch_size=3,
    image_size=(500, 500)
)

validation_ds = id.image_dataset_from_directory(
    directory='C:/Users/ryder/Downloads/Supernatural_Validation_Dataset/',
    labels='inferred',
    label_mode='categorical',
    batch_size=3,
    image_size=(500, 500)
)

model = keras.applications.Xception(weights=None, input_shape=(500, 500, 3), classes=4)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(training_data, epochs=10, validation_data=validation_ds)
print(model.accuracy)
# import dataset from onedrive folder
'''
# Loading the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
output_classes = 10

# Scale images to the [0, 1] range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Make sure images have shape (28, 28, 1)
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

## Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(output_classes, activation="sigmoid"),
    ]
)

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=128)

## Evaluate the trained model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\n Test accuracy:', test_acc)'''
