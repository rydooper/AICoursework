import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import load_model
import keras_tuner as kt
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# this must be set before importing tf to avoid displaying warnings in console
tf.compat.v1.disable_eager_execution()

classNames = ['Cas', 'Dean', 'Sam']
trainDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/' \
           r'supernatural-images-dataset/Supernatural_Train_Dataset/'
validDir = r'C:/Users/ryder/OneDrive/Documents/GitHub-Laptop/' \
           r'supernatural-images-dataset/Supernatural_Validation_Dataset/'


def runModel():
    def model_builder(hp):
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)

        model = keras.Sequential(
            [
                keras.Input(shape=(300, 300)),
                keras.layers.Flatten(),
                keras.layers.Dense(units=hp_units, activation='relu'),
                keras.layers.Dense(10)
            ]
        )
        #model.add(keras.layers.Flatten(input_shape=(300, 300)))
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    trainDatagen = ImageDataGenerator(
        rescale=1. / 300,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    testDatagen = ImageDataGenerator(rescale=1. / 300)
    trainGenerator = trainDatagen.flow_from_directory(
        directory=trainDir,
        target_size=(300, 300),
        class_mode="categorical",
    )

    validGenerator = testDatagen.flow_from_directory(
        directory=validDir,
        target_size=(300, 300),
        class_mode="categorical",
    )

    tuner = kt.Hyperband(model_builder,
                         objective='accuracy',
                         max_epochs=10,
                         factor=3)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    tuner.search(trainGenerator, epochs=50, callbacks=[stop_early])
    # tuner.search(trainGenerator, classNames, epochs=2, validation_split=0.2)

    # Get the optimal hyperparams
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparams and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(trainGenerator, classNames, epochs=50, validation_split=0.2)
    acc_per_epoch = history.history['accuracy']
    best_epoch = acc_per_epoch.index(max(acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(trainGenerator, classNames, epochs=best_epoch, validation_split=0.2)
    eval_result = hypermodel.evaluate(validGenerator, classNames)
    print("[test loss, test accuracy]:", eval_result)


if __name__ == '__main__':
    runModel()

'''tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
)'''

