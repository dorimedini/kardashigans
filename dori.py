# Imports
from keras import backend as K
import numpy as np
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Input, Dropout, Concatenate, Flatten
from keras.utils import to_categorical, Sequence, plot_model
from keras.models import Model, Sequential
from keras.layers.merge import add
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16

# Drive mounting
from google.colab import drive
drive.mount('/content/gdrive')


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


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_weights_only=False,
                 period=[]):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period

    def on_train_begin(self, logs=None):
        filepath = self.filepath.format(epoch="start", **logs)
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)
        print("saved model to {}".format(filepath))

    def on_epoch_end(self, epoch, logs=None):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if epoch in self.period:
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
            print("saved model to {}".format(filepath))

    def on_train_end(self, logs=None):
        filepath = self.filepath.format(epoch="end", **logs)
        if self.save_weights_only:
            self.model.save_weights(filepath, overwrite=True)
        else:
            self.model.save(filepath, overwrite=True)
        print("saved model to {}".format(filepath))


# Given a dataset, constructs a model with the requested parameters
# and runs it after freezing layers. Also, optionally uses specified
# weights.
#
# The dataset object should have a load_data() method (named in
# keras.datasets).
#
# The shape parameter should be (28**2,) for mnist and (32**2,)
# for cifur10.
#
# Including input and output layers, we'll have a total of n_layers+2
# layers in the resulting topology.
#
# The layers_to_freeze parameter expects a list of layer indexes
# in the range 0,1,2...n_layers+1 (yes, plus one, we're counting
# input and output layers as well).
#
# The weight_map, if defined, should map layer indexes (in the
# range [0,n_layers+1]) to weights (as output by layer.get_weights()).
def frozen_dense_fit(dataset,
                     shape,
                     n_layers=3,
                     n_classes=10,
                     n_neurons=256,
                     epochs=100,
                     batch_size=32,
                     activation='relu',
                     output_activation='softmax',
                     layers_to_freeze=[],
                     weight_map={}):

    # Load dataset, give the dimensions expected by keras
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # Construct layers, freeze / set weights if requested.
    # Input layer gets special treatment:
    input_layer = Input(shape=shape)
    if 0 in weight_map:
        input_layer.set_weights(weight_map[0])
    layers = [input_layer]
    # Now the rest of the scum:
    for i in range(n_layers):
        if i == n_layers-1:  # Output layer also gets special treatment
            layer = Dense(n_classes, activation=output_activation)
        else:
            layer = Dense(n_neurons, activation=activation)
        if i in layers_to_freeze:
            layer.trainable = False
        if i in weight_map:
            layer.set_weights(weight_map[i])
        layers += [layer]

    # Connect first layer to input, then connect each subsequent
    # layer to the previous connected layer.
    input_layer = layers[0]
    next_layer = layers[1]  # Could be output
    connected_layers = [next_layer(input_layer)]
    for i in range(2, len(layers)):
        current_layer = layers[i]
        last_connected = connected_layers[-1]
        connected_layers += [current_layer(last_connected)]
    output_layer = connected_layers[-1]
    model = Model(input_layer, output_layer)

    # For 100 epochs we'll get checkpoints at epochs 0, 1, 2, 3, 8, 40, 90, 100.
    # For N epochs we'll get checkpoints at 0, 0.01N, 0.02N, 0.03N, 0.08N, 0.4N,
    # 0.9N, N (rounded down)
    checkpoint_epochs = [(percentile * epochs) // 100 for percentile in [0, 1, 2, 3, 8, 40, 90, 100]]
    filepath = "vanilla.{epoch}.hdf5"
    checkpoint_cb = CustomModelCheckpoint(filepath, period=checkpoint_epochs)

    # That's a wrap. Compile & train
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train,
              shuffle=True,
              epochs=epochs,
              callbacks=[checkpoint_cb],
              batch_size=batch_size,
              validation_data=(x_test, y_test))
    return model


def sandbox():
    model = frozen_dense_fit(dataset=mnist,
                             shape=(28 ** 2,))
    old_layer_weights = [layer.get_weights() for layer in model.layers]
    print("============================================================")
    print("Model has {} layers (should be {})".format(len(model.layers), len(old_layer_weights)))
    print("Weight lists are of length: {}".format([len(layer_weights) for layer_weights in old_layer_weights]))
    old_l1_weight0 = old_layer_weights[1][0][0][0]
    old_l2_weight0 = old_layer_weights[2][0][0][0]
    print("=================== OLD LAYER1 FIRST WEIGHT: ====================")
    print(old_l1_weight0)
    print("=================== OLD LAYER2 FIRST WEIGHT: ====================")
    print(old_l2_weight0)
    print("=============== FREEZE LAYER1, SEE WHAT CHANGES =================")
    model = frozen_dense_fit(dataset=mnist,
                             shape=(28 ** 2,),
                             layers_to_freeze=[1],
                             weight_map={1: old_layer_weights[1]})
    new_l1_weight0 = model.layers[1].get_weights()[0][0][0]
    new_l2_weight0 = model.layers[2].get_weights()[0][0][0]
    print("=================== NEW LAYER1 FIRST WEIGHT: ====================")
    print(new_l1_weight0)
    print("=================== NEW LAYER2 FIRST WEIGHT: ====================")
    print(new_l2_weight0)
    print("============================================================")
    print(".")
    print(".")
    print(".")
    if new_l1_weight0 == old_l1_weight0 and new_l2_weight0 != old_l2_weight0:
        print("OK")
    else:
        print("FAIL")
        if new_l1_weight0 != old_l1_weight0:
            print("Layer #1s first weight should have stayed {}, is now {}".format(old_l1_weight0, new_l1_weight0))
        else:
            print("Layer #1s first weight didn't change but neither did layer #2s first weight... both {}".format(
                new_l2_weight0))


if __name__ == '__main__':
    sandbox()
