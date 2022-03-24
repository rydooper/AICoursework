# CNN
from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# this must be set before importing tf to avoid displaying warnings in console
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.disable_eager_execution()


def runModel():
    output_classes = 3
    classNames = ['Cas', 'Dean', 'Sam']

    trainDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/' \
               r'supernatural-images-dataset/Supernatural_Train_Dataset/'
    validDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/' \
               r'supernatural-images-dataset/Supernatural_Validation_Dataset/'

    trainDatagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    testDatagen = ImageDataGenerator(rescale=1. / 255)
    trainGenerator = trainDatagen.flow_from_directory(
        directory=trainDir,
        target_size=(300, 300),
        color_mode="rgb",
        classes=classNames,
        batch_size=10,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    validGenerator = testDatagen.flow_from_directory(
        directory=validDir,
        target_size=(300, 300),
        color_mode="rgb",
        classes=classNames,
        batch_size=10,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    model = keras.Sequential(
        [
            keras.Input(shape=(300, 300, 3)),
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

    stepSizeTrain = trainGenerator.n // trainGenerator.batch_size
    stepSizeValid = validGenerator.n // validGenerator.batch_size
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=trainGenerator,
                        steps_per_epoch=stepSizeTrain,
                        validation_data=validGenerator,
                        validation_steps=stepSizeValid,
                        epochs=10)
    model.evaluate(x=trainGenerator,
                   steps=stepSizeValid)

    model.save('supernaturalCNNModel.h5')


if __name__ == '__main__':
    runModel()
