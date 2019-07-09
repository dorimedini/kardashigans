# Imports
import numpy as np
from inspect import getframeinfo, stack
from keras import backend as K
from keras import optimizers
from keras.datasets import mnist, cifar10
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten
from keras.utils import to_categorical, Sequence, plot_model
from keras.layers.merge import add
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

"""
Functions
"""


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


def get_layers_weights(model):
    return [layer.get_weights() for layer in model.layers]


def calc_robustness(test_data, model, source_weights_model=None, layer_indices=[], batch_size=32):
    """
    Evaluates the model on test data.

    Optionally, if source_weights_model is given, sets the weights
    of the model (for layer indices appearing in layer_indices) at
    each given layer to the weights of source_weights_model.

    :param test_data: A tuple (x_test, y_test) for validation
    :param model: The model to evaluate
    :param source_weights_model: The model from which we should
        copy weights and update our model's weights before eval.
    :param layer_indices: Layers to reset the weights of.
    :param batch_size: Self explanatory
    :return: A number in the interval [0,1] representing accuracy.
    """
    x_test, y_test = test_data
    if source_weights_model:
        for idx in layer_indices:
            loaded_weights = source_weights_model.layers[idx].get_weights()
            model.layers[idx].set_weights(loaded_weights)
    evaluated_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
    return evaluated_metrics[model.metrics_names.index('acc')]


def reset_to_checkpoint(model, checkpoint_weights):
    for idx in range(len(checkpoint_weights)):
        model.layers[idx].set_weights(checkpoint_weights[idx])


"""
Classes
"""


class Verbose(object):
    def __init__(self, verbose=False):
        self._verbose = verbose

    def _print(self, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        if self._verbose:
            print("%s:%d - %s" % (caller.filename, caller.lineno, args[0]), *args[1:], **kwargs)


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
    def __init__(self, filepath_template, period=[], verbose=False):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath_template = filepath_template
        self.period = period
        self._printer = Verbose(verbose=verbose)

    def on_train_begin(self, logs=None):
        filepath = self.filepath_template.format(epoch="start", **logs)
        self.model.save(filepath, overwrite=True)
        self._printer._print("saved model to {}".format(filepath))

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.period:
            filepath = self.filepath_template.format(epoch=epoch, **logs)
            self.model.save(filepath, overwrite=True)
            self._printer._print("saved model to {}".format(filepath))

    def on_train_end(self, logs=None):
        filepath = self.filepath_template.format(epoch="end", **logs)
        self.model.save(filepath, overwrite=True)
        self._printer._print("saved model to {}".format(filepath))
