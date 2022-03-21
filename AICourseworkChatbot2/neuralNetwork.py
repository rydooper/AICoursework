# CNN
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

classNames = ['Cas', 'DW', 'SW']


def runModel():
    output_classes = 3

    trainDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/supernatural-images-dataset/Supernatural_Train_Dataset/'
    validDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/supernatural-images-dataset/Supernatural_Validation_Dataset/'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=trainDir,
        target_size=(500, 500),
        color_mode="rgb",
        classes=classNames,
        batch_size=10,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    valid_generator = test_datagen.flow_from_directory(
        directory=validDir,
        target_size=(500, 500),
        color_mode="rgb",
        classes=classNames,
        batch_size=10,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

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

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=10
                        )
    model.evaluate(x=train_generator,
                   steps=STEP_SIZE_VALID)

    model.save('supernaturalSWDWCas_Model.h5')


if __name__ == '__main__':
    runModel()
