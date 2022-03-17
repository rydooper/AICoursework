# CNN
import kwargs
import numpy as np
import tensorflow as tf
from tensorflow import keras

training_DS = keras.preprocessing.image_dataset_from_directory(
    directory="C:/Users/ryder/Downloads/Supernatural_Train_Dataset/",
    labels='inferred',
    label_mode='categorical',
    batch_size=10,
    image_size=(500, 500),
    shuffle=False,
    subset="training",
    seed=3,
    interpolation="bilinear"
)
print(training_DS.element_spec)
# training_DS.element_spec = tf.TensorShape([500, 500, 4])

validation_DS = keras.preprocessing.image_dataset_from_directory(
    directory='C:/Users/ryder/Downloads/Supernatural_Train_Dataset/',
    labels='inferred',
    label_mode='categorical',
    batch_size=10,
    image_size=(500, 500),
    validation_split=0.2,
    shuffle=False,
    subset="validation",
    seed=3,
    interpolation="bilinear"
)

# validation_split = 0.2 eg
# subset = validation
# a seed


output_classes = 3
model = keras.Sequential(
    [
        keras.Input(shape=(500, 500, 3)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(output_classes, activation="sigmoid"),
    ]
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(training_DS, epochs=10, validation_data=validation_DS)
model.fit(training_DS, epochs=10, validation_data=validation_DS, batch_size=10)
# print(model.accuracy)
model.save('supernaturalSWDWCas_Model.h5')
