from datetime import datetime
from keras import optimizers
from keras.datasets import mnist, cifar10
from keras.models import model_from_json
import os
import pytz
from kardashigans.experiment import Experiment, ExperimentWithCheckpoints
from kardashigans.analyze_model import AnalyzeModel
from kardashigans.trainer import FCTrainer
from kardashigans.resource_manager import ResourceManager


class Baseline(ExperimentWithCheckpoints):
    """
    Train MNIST and CIFAR10 on FC nets, 3 layers, and try to reproduce results
    on robustness of specific layers.

    Phase1 checks each layer of each model when we reset the layer to initial (pre-
    train) weights, and phase2 re-initializes each layer to specific weight checkpoints
    (by epoch) when testing robustness.
    """

    def __init__(self,
                 root_dir='/content/drive/My Drive/globi/',
                 resource_load_dir=None):
        mnist_name = Baseline.get_model_name(Experiment.get_dataset_name(mnist))
        cifar10_name = Baseline.get_model_name(Experiment.get_dataset_name(cifar10))
        super(Baseline, self).__init__(name='Baseline',
                                       model_names=[mnist_name, cifar10_name],
                                       trainers={
                                           mnist_name: Baseline.construct_dataset_trainer(mnist),
                                           cifar10_name: Baseline.construct_dataset_trainer(cifar10)
                                       },
                                       root_dir=root_dir,
                                       period=Baseline.get_epoch_save_period(),
                                       resource_load_dir=resource_load_dir)
        self._dataset_names = [Experiment.get_dataset_name(dataset) for dataset in [mnist, cifar10]]

    @staticmethod
    def get_epoch_save_period():
        return [0, 1, 2, 3, 8, 40, 90]

    def get_checkpoint_epoch_keys(self):
        return ResourceManager.get_checkpoint_epoch_keys(Baseline.get_epoch_save_period())

    @staticmethod
    def get_model_name(dataset_name):
        return "{dataset}_fc{layers}_batch{batch}_epochs{epochs}_{opt}" \
               "".format(dataset=dataset_name,
                         layers=Baseline.get_dataset_n_layers(dataset_name),
                         batch=Baseline.get_dataset_batch_size(dataset_name),
                         epochs=Baseline.get_dataset_n_epochs(dataset_name),
                         opt=Baseline.get_dataset_optimizer_string(dataset_name))

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
    def get_dataset_n_parameter_layers(dataset_name):
        """ N hidden layers + input/output layers = N+1 parameter layers """
        return Baseline.get_dataset_n_layers(dataset_name) + 1

    @staticmethod
    def get_dataset_optimizer(dataset_name):
        if dataset_name == 'mnist':
            return optimizers.SGD(momentum=0.9, nesterov=True)
        elif dataset_name == 'cifar10':
            return optimizers.SGD(momentum=0.9, nesterov=True)

    @staticmethod
    def get_dataset_optimizer_string(dataset_name):
        optimizer = Baseline.get_dataset_optimizer(dataset_name)
        params = dict(optimizer.get_config().items())
        return "lr{lr:.3f}_momentum{momentum:.2f}_decay{decay:.2f}_nesterov{nesterov}" \
               "".format(lr=params['lr'],
                         momentum=params['momentum'],
                         decay=params['decay'],
                         nesterov='TRUE' if params['nesterov'] else 'FALSE')

    @staticmethod
    def get_dataset_batch_size(dataset_name):
        if dataset_name == 'mnist':
            return 32
        elif dataset_name == 'cifar10':
            return 128

    @staticmethod
    def construct_dataset_trainer(dataset):
        dataset_name = Experiment.get_dataset_name(dataset)
        return FCTrainer(dataset=dataset,
                         epochs=Baseline.get_dataset_n_epochs(dataset_name),
                         n_layers=Baseline.get_dataset_n_layers(dataset_name),
                         batch_size=Baseline.get_dataset_batch_size(dataset_name),
                         optimizer=Baseline.get_dataset_optimizer(dataset_name))

    def _phase1_dataset_robustness(self, dataset_name):
        model_name = Baseline.get_model_name(dataset_name)
        test_set = self._test_sets[model_name]
        with self.open_model(model_name) as model:
            clean_results = AnalyzeModel.calc_robustness(test_data=(test_set['x'], test_set['y']),
                                                         model=model,
                                                         batch_size=Baseline.get_dataset_batch_size(dataset_name))
            try:
                with self.open_model_at_epoch(model_name, 'start') as start_model:
                    robustness = [AnalyzeModel.calc_robustness(test_data=(test_set['x'], test_set['y']),
                                                               model=model,
                                                               source_weights_model=start_model,
                                                               layer_indices=[i],
                                                               batch_size=Baseline.get_dataset_batch_size(dataset_name))
                                  for i in range(len(model.layers))]
            except Exception as e:
                self.logger.error("Missing 'start' checkpoint in phase1, cannot continue.")
                self.logger.error("Exception: {}".format(e))
                return [clean_results] + [0 for i in range(len(model.layers))]
        robustness = [clean_results] + robustness
        self.logger.debug("{} robustness: by layer: {}".format(model_name, robustness))
        return robustness

    # Phase 1: Train, pick a layer(s), re-init to random and evaluate
    # (evaluate == Check robustness)
    def phase1(self):
        data = []
        rows = []
        for dataset_name in self._dataset_names:
            data += [self._phase1_dataset_robustness(dataset_name)]
            rows += [Baseline.get_model_name(dataset_name)]
        self.logger.debug("Robustness results: got {} rows, with {} columns on the first row, "
                          "row labels are {}".format(len(data), len(data[0]), rows))
        n_layers = len(data[0]) - 1
        AnalyzeModel.generate_heatmap(data=data,
                                      row_labels=rows,
                                      col_labels=["Baseline"] + ["Layer %d" % i for i in range(n_layers)],
                                      filename="phase1_heatmap.png",
                                      output_dir=self._results_dir)

    def _phase2_dataset_robustness_by_epoch(self, model, dataset_name, layer):
        model_name = Baseline.get_model_name(dataset_name)
        checkpoints = self.get_checkpoint_epoch_keys()
        test_set = self._test_sets[model_name]
        robustness = []
        clean_results = AnalyzeModel.calc_robustness(
            test_data=(test_set['x'], test_set['y']),
            model=model,
            batch_size=Baseline.get_dataset_batch_size(dataset_name))
        for epoch in checkpoints:
            try:
                with self.open_model_at_epoch(model_name, epoch) as checkpoint_model:
                    robustness += [AnalyzeModel.calc_robustness(test_data=(test_set['x'], test_set['y']),
                                                                model=model,
                                                                source_weights_model=checkpoint_model,
                                                                layer_indices=[layer],
                                                                batch_size=Baseline.get_dataset_batch_size(
                                                                    dataset_name))]
            except Exception as e:
                self.logger.error("Missing checkpoint at epoch {} in phase2, cannot continue".format(epoch))
                self.logger.error("Exception: {}".format(e))
                return [clean_results] + [0 for i in range(len(checkpoints))]
        robustness = [clean_results] + robustness
        self.logger.debug("{} robustness of layer {} by epoch: {}".format(model_name, layer, robustness))
        return robustness

    # Phase 2: Train, pick a layer, re-init to specific epochs, evaluate
    def phase2(self):
        # Output a separate heatmap for each dataset.
        # Rows are layers, columns are epochs from which the weights were
        # taken.
        epochs = self.get_checkpoint_epoch_keys()
        for dataset_name in self._dataset_names:
            self.logger.debug("Running phase2 on {}".format(dataset_name))
            data = []
            rows = []
            model_name = Baseline.get_model_name(dataset_name)
            with self.open_model(model_name) as model:
                for layer in range(len(model.layers)):
                    data += [self._phase2_dataset_robustness_by_epoch(model, dataset_name, layer)]
                    rows += ["Layer {}".format(layer)]
            AnalyzeModel.generate_heatmap(data=data,
                                          row_labels=rows,
                                          col_labels=["Baseline"] + ["Epoch {}".format(e) for e in epochs],
                                          filename="phase2_{}_heatmap.png".format(dataset_name),
                                          output_dir=self._results_dir)

    def go(self):
        self.phase1()
        self.phase2()
