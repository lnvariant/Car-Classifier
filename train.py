import keras
from livelossplot.keras import PlotLossesCallback

# Python File based imports (do not add these to notebooks)
from dataloader import get_generators, get_data_frames
from model import MAIN_DIR_PATH, get_model


def train_model(training_generator, validation_generator, model_name="xception_model", augment=False, lr=1e-4,
                epochs=10, early_stop=False, fine_tune_from=-1, model=None):
    """
    Returns a trained model and the training history.

    :param training_generator: the training data generator
    :param validation_generator: the validation data generator
    :param augment: whether or not the augment the data
    :param model_name: name of the new model (weights will be saved under this name)
    :param lr: learning rate for model
    :param epochs: number of epochs to run for
    :param early_stop: whether or not to stop early if the validation and training curves diverge too much
    :param fine_tune_from: which layer to start fine tuning from (all layers before this layer will be frozen)
    :param model: an existing model to train
    """

    if model is None:
        model = get_model(15)

    # Fine Tuning
    if fine_tune_from >= 0:
        print(len(model.layers))

        for layer in model.layers:
            layer.trainable = False

        for layer in model.layers[fine_tune_from:]:
            layer.trainable = True

    model.compile(optimizer="nadam",
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    # Setup training callbacks
    callbacks = []
    if early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=50))
    callbacks.append(keras.callbacks.ModelCheckpoint(MAIN_DIR_PATH + "/" + model_name + ".h5", save_best_only=True))
    callbacks.append(PlotLossesCallback())

    model.summary()

    train_step_size = training_generator.n // training_generator.batch_size
    history = model.fit_generator(training_generator, validation_data=validation_generator,
                                  epochs=epochs, steps_per_epoch=train_step_size,
                                  callbacks=callbacks)

    return model, history


def main():
    training_df, validation_df = get_data_frames(MAIN_DIR_PATH + "data/car_training_data.csv",
                                                 MAIN_DIR_PATH + "data/car_validation_data.csv")
    training_generator, validation_generator = get_generators("", training_df, validation_df)

    # loaded_model = load_model(MAIN_DIR_PATH + "make_model2.h5")
    model, history = train_model(training_generator, validation_generator, model_name="car_model", augment=False,
                                 lr=1e-3, epochs=20, early_stop=True)


