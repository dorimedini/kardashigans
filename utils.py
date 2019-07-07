# Imports
from keras import backend as K
from keras import optimizers
import numpy as np
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten
from keras.utils import to_categorical, Sequence, plot_model
from keras.layers.merge import add
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

# Drive mounting
# from google.colab import drive
# drive.mount('/content/drive')
# Define the working directory everyone should use
ROOT_DIR = '/content/drive/My Drive/globi/'

"""
Functions
"""


def copy_layers(target, source, layers):
    assert len(target.layers) == len(source.layers)
    assert max(layers) < len(target.layers)
    for idx in layers:
        target.layers[idx].set_weights(source.layers[idx].get_weights())


def reset_layers(model, layers):
    assert max(layers) < len(model.layers)
    session = K.get_session()
    for idx in layers:
        layer = model.layers[idx]
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)


def get_dataset_name(dataset):
    name = dataset.__name__
    return name[name.rfind(".") + 1:]


def get_epoch_checkpoints():
    return [0, 1, 2, 3, 8, 40, 90, 100]


def save_model(model, filepath, weights_only=False):
    if weights_only:
        model.save_weights(filepath, overwrite=True)
    else:
        model.save(filepath, overwrite=True)
# load_model is just imported from keras.models


def get_layers_weights(model):
    return [layer.get_weights() for layer in model.layers]


def calc_robustness(test_data, model, epochs_weights, layer_indices=[], batch_size=32):
    results = {}
    x_test, y_test = test_data
    results["baseline"] = model.evaluate(x_test, y_test, batch_size=batch_size)
    for epoch, weights in epochs_weights.items():
        for idx in layer_indices:
            model.layers[idx].set_weights(weights[idx])
        results[epoch] = model.evaluate(x_test, y_test, batch_size=batch_size)
    return results


def reset_to_checkpoint(model, checkpoint_weights):
    for idx in range(len(checkpoint_weights)):
        model.layers[idx].set_weights(checkpoint_weights[idx])


"""
Callbacks
"""


class SaveEpochsWeightsToDictCheckpoint(Callback):
    def __init__(self, weights_dict, period=[]):
        super(SaveEpochsWeightsToDictCheckpoint, self).__init__()
        self.weights_dict = weights_dict
        self.period = period

    def on_train_begin(self, logs=None):
        self.weights_dict["train_begin"] = get_layers_weights(self.model)

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            self.weights_dict[str(epoch)] = get_layers_weights(self.model)

    def on_train_end(self, logs=None):
        self.weights_dict["train_end"] = get_layers_weights(self.model)


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_weights_only=False, period=None):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period if period else []

    def on_train_begin(self, logs=None):
        filepath = self.filepath.format(epoch="start", **logs)
        save_model(self.model, filepath, self.save_weights_only)
        print("saved model to {}".format(filepath))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            filepath = self.filepath.format(epoch=epoch, **logs)
            save_model(self.model, filepath, self.save_weights_only)
            print("saved model to {}".format(filepath))

    def on_train_end(self, logs=None):
        filepath = self.filepath.format(epoch="end", **logs)
        save_model(self.model, filepath, self.save_weights_only)
        print("saved model to {}".format(filepath))
