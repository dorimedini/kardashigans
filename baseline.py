from datetime import datetime
from experiment import ExperimentWithCheckpoints
from keras import optimizers
from keras.models import model_from_json
from trainer import FCTrainer
import matplotlib.pylab as plt
import os
import pytz
import random
import seaborn as sns
import utils as U


class Baseline(ExperimentWithCheckpoints):
    """
    Train MNIST and CIFAR10 on FC nets, 3 layers, and try to reproduce results
    on robustness of specific layers.

    Phase1 checks each layer of each model when we reset the layer to initial (pre-
    train) weights, and phase2 re-initializes each layer to specific weight checkpoints
    (by epoch) when testing robustness.
    """
    def __init__(self, resource_load_dir=None, verbose=False):
        # Map model names to dataset names on which they run ('phase1_mnist' -> 'mnist')
        random.seed()  # Won't need this when we replace the stub _check_robustness
        super(Baseline, self).__init__(name='Baseline',
                                       model_names=['mnist', 'cifar10'],
                                       verbose=verbose,
                                       trainers={
                                           'mnist': Baseline.construct_dataset_trainer(U.mnist, verbose),
                                           'cifar10': Baseline.construct_dataset_trainer(U.cifar10, verbose)
                                       },
                                       resource_load_dir=resource_load_dir)

    # TODO: Implement this method
    def _check_robustness(self, model, layers=None, init_from_epoch=None):
        return random.random()

    @staticmethod
    def get_dataset_n_epochs(dataset_name):
        if dataset_name == 'mnist':
            return 100
        elif dataset_name == 'cifar10':
            return 100

    @staticmethod
    def get_dataset_n_layers(dataset_name):
        if dataset_name == 'mnist':
            return 3
        elif dataset_name == 'cifar10':
            return 3

    @staticmethod
    def get_dataset_optimizer(dataset_name):
        if dataset_name == 'mnist':
            return optimizers.SGD(momentum=0.9, nesterov=True)
        elif dataset_name == 'cifar10':
            return optimizers.SGD(momentum=0.9, nesterov=True)

    @staticmethod
    def get_dataset_batch_size(dataset_name):
        if dataset_name == 'mnist':
            return 32
        elif dataset_name == 'cifar10':
            return 128

    @staticmethod
    def construct_dataset_trainer(dataset, verbose=False):
        dataset_name = U.get_dataset_name(dataset)
        return FCTrainer(dataset=dataset,
                         verbose=verbose,
                         epochs=Baseline.get_dataset_n_epochs(dataset_name),
                         n_layers=Baseline.get_dataset_n_layers(dataset_name),
                         batch_size=Baseline.get_dataset_batch_size(dataset_name),
                         optimizer=Baseline.get_dataset_optimizer(dataset_name))

    def _phase1_dataset_robustness(self, model_name):
        model = self._dataset_fit(model_name)
        robustness = [self._check_robustness(model, [i]) for i in range(len(model.layers))]
        self._print("{} robustness: by layer: {}".format(model_name, robustness))
        return robustness

    # Phase 1: Train, pick a layer(s), re-init to random and evaluate
    # (evaluate == Check robustness)
    def phase1(self):
        data = []
        rows = []
        for model_name in self._model_names:
            data += [self._phase1_dataset_robustness(model_name)]
            rows += [model_name]
        self._print("Robustness results: got {} rows, with {} columns on the first row, "
                    "row labels are {}".format(len(data), len(data[0]), rows))
        self.generate_heatmap(data=data,
                              row_labels=rows,
                              col_labels=["Layer %d" % i for i in range(len(data[0]))],
                              filename="phase1_heatmap.png")

    def _phase2_dataset_robustness_by_epoch(self, model_name, layer):
        checkpoints = self._trainers[model_name].get_epoch_checkpoints()
        model = self._dataset_fit(model_name)
        robustness = [self._check_robustness(model, [layer], epoch) for epoch in checkpoints]
        self._print("{} robustness of layer {} by epoch: {}".format(model_name, layer, robustness))
        return robustness

    # Phase 2: Train, pick a layer, re-init to specific epochs, evaluate
    def phase2(self):
        # Output a separate heatmap for each dataset.
        # Rows are layers, columns are epochs from which the weights were
        # taken.
        for model_name in self._model_names:
            self._print("Running phase2 on {}".format(model_name))
            data = []
            rows = []
            for layer in range(Baseline.get_dataset_n_layers(model_name)):
                data += [self._phase2_dataset_robustness_by_epoch(model_name, layer)]
                rows += ["Layer {}".format(layer)]
            self.generate_heatmap(data=data,
                                  row_labels=rows,
                                  col_labels=["Epoch {}".format(e) for e in U.get_epoch_checkpoints()],
                                  filename="phase2_{}_heatmap.png".format(model_name))

    def go(self):
        self.phase1()
        self.phase2()


if __name__ == "__main__":
    baseline = Baseline(verbose=True)
    baseline.go()
