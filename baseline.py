from datetime import datetime
from keras import optimizers
from keras.models import model_from_json
import matplotlib.pylab as plt
import os
import pytz
import random
import seaborn as sns
import utils


# Baseline results: CIFAR and MNIST on 4-layer FC nets.
# If only_on_datasets contains a list of datasets, then
# the baseline results are only tested on those datasets.
# TODO: Allow load from existing model
class Baseline:
    def __init__(self,
                 verbose=False,
                 only_on_datasets=[utils.mnist, utils.cifar10],
                 root_dir=''):
        random.seed()  # Won't need this when we replace the stub _check_robustness
        self._verbose = verbose
        self._datasets = only_on_datasets
        self._trained_datasets = {}
        self._setup_env(root_dir)

    # Simple filtered print
    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    # TODO: Implement this method
    def _check_robustness(self, model, layers=None, init_from_epoch=None):
        return random.random()

    def _setup_env(self, root_dir):
        self._base_dir = root_dir + "baseline/"
        self._time_started = datetime.now(pytz.timezone('Israel')).strftime("%d_%m_%Y___%H_%M_%S")
        self._run_dir = self._base_dir + self._time_started + "/"
        self._results_dir = self._run_dir + "RESULTS/"
        self._models_dir = self._run_dir + "MODELS/"
        self._print("In _setup_env(), setting up test dir at {}".format(self._run_dir))
        if not os.path.isdir(self._base_dir):
            os.mkdir(self._base_dir)
        if not os.path.isdir(self._run_dir):
            os.mkdir(self._run_dir)
        if not os.path.isdir(self._results_dir):
            os.mkdir(self._results_dir)
        if not os.path.isdir(self._models_dir):
            os.mkdir(self._models_dir)
        self._print("Test dir setup complete")

    def _save_model(self, model, name):
        # TODO: Use Gal's stuff
        model_prefix = self._models_dir + name
        with open(model_prefix + ".json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(model_prefix + ".h5")

    def _load_model(self, name):
        # TODO: Use Gal's stuff
        model_prefix = self._models_dir + name
        with open(model_prefix + ".json", 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_prefix + ".h5")

    def _generate_heatmap(self, data, row_labels, col_labels, filename):
        self._print("Generating heatmap. Data: {}".format(data))
        self._print("Rows: {}".format(row_labels))
        self._print("Cols: {}".format(col_labels))
        ax = sns.heatmap(data, linewidth=0.5, xticklabels=col_labels, yticklabels=row_labels)
        fig = ax.get_figure()
        fig.savefig(self._results_dir + filename)
        if self._verbose:
            plt.show()

    @staticmethod
    def get_dataset_n_epochs(dataset):
        name = utils.get_dataset_name(dataset)
        if name == 'mnist':
            return 100
        elif name == 'cifar10':
            return 1000

    @staticmethod
    def get_dataset_n_layers(dataset):
        name = utils.get_dataset_name(dataset)
        if name == 'mnist':
            return 3
        elif name == 'cifar10':
            return 3

    @staticmethod
    def get_dataset_optimizer(dataset):
        name = utils.get_dataset_name(dataset)
        if name == 'mnist':
            return optimizers.SGD(momentum=0.9, nesterov=True)
        elif name == 'cifar10':
            return optimizers.SGD(momentum=0.9, nesterov=True)

    @staticmethod
    def get_dataset_batch_size(dataset):
        name = utils.get_dataset_name(dataset)
        if name == 'mnist':
            return 32
        elif name == 'cifar10':
            return 128

    def _dataset_fit(self, dataset):
        dataset_name = utils.get_dataset_name(dataset)
        # Return previously trained model, no need to train twice
        if dataset_name not in self._trained_datasets:
            self._print("Fitting dataset {}".format(dataset_name))
            fdf = utils.FrozenDenseFit(dataset=dataset,
                                       verbose=self._verbose,
                                       epochs=Baseline.get_dataset_n_epochs(dataset),
                                       n_layers=Baseline.get_dataset_n_layers(dataset),
                                       batch_size=Baseline.get_dataset_batch_size(dataset),
                                       optimizer=Baseline.get_dataset_optimizer(dataset))
            model = fdf.go()
            self._trained_datasets[dataset_name] = model
        else:
            self._print("Already trained model for {}, returning it".format(dataset_name))
        return self._trained_datasets[dataset_name]

    def _phase1_dataset_robustness(self, dataset):
        dataset_name = utils.get_dataset_name(dataset)
        model = self._dataset_fit(dataset)
        self._save_model(model, "phase1_" + dataset_name)
        robustness = [self._check_robustness(model, [i]) for i in range(len(model.layers))]
        self._print(dataset_name + " robustness: by layer: {}".format(robustness))
        return robustness

    # Phase 1: Train, pick a layer(s), re-init to random and evaluate
    # (evaluate == Check robustness)
    def phase1(self):
        data = []
        rows = []
        for dataset in self._datasets:
            data += [self._phase1_dataset_robustness(dataset)]
            rows += [utils.get_dataset_name(dataset)]
        self._print("Robustness results: got {} rows, with {} columns on the first row, "
                    "row labels are {}".format(len(data), len(data[0]), rows))
        # Make sure the data rows have the same number of columns
        # (i.e, make sure each model had the same number of layers)
        if len(data) > 1:
            n_cols = len(data[0])
            for i in range(1, len(data)):
                assert n_cols == len(data[i]), \
                    "All dataset robustness results must have same size (different net " \
                    "topologies used by accident?), currently {} has {} robustness " \
                    "tests and {} has {}".format(utils.get_dataset_name(self._datasets[0]),
                                                 n_cols,
                                                 utils.get_dataset_name(self._datasets[i]),
                                                 len(data[i]))

        self._generate_heatmap(data=data,
                               row_labels=rows,
                               col_labels=["Layer %d" % i for i in range(len(data[0]))],
                               filename="phase1_heatmap.png")

    def _phase2_dataset_robustness_by_epoch(self, dataset, layer):
        checkpoints = utils.get_epoch_checkpoints()
        model = self._dataset_fit(dataset)
        dataset_name = utils.get_dataset_name(dataset)
        self._save_model(model, "phase2_" + dataset_name)
        robustness = [self._check_robustness(model, [layer], epoch) for epoch in checkpoints]
        self._print(dataset_name + " robustness of layer {} by epoch: {}".format(layer, robustness))
        return robustness

    # Phase 2: Train, pick a layer, re-init to specific epochs, evaluate
    def phase2(self):
        # Output a separate heatmap for each dataset.
        # Rows are layers, columns are epochs from which the weights were
        # taken.
        for dataset in self._datasets:
            name = utils.get_dataset_name(dataset)
            self._print("Running phase2 on {}".format(name))
            data = []
            rows = []
            for layer in range(Baseline.get_dataset_n_layers(dataset)):
                data += [self._phase2_dataset_robustness_by_epoch(dataset, layer)]
                rows += ["Layer {}".format(layer)]
            n_cols = len(data[0])
            for i in range(1, len(data)):
                assert n_cols == len(data[i]), "Different number of epoch checkpoints " \
                                               "on different layers...? n_cols == {} but" \
                                               "len(data[{}]) == {}".format(n_cols, i, len(data[i]))
            self._generate_heatmap(data=data,
                                   row_labels=rows,
                                   col_labels=["Epoch {}".format(e) for e in utils.get_epoch_checkpoints()],
                                   filename="phase2_{}_heatmap.png".format(name))

    def go(self):
        assert len(self._datasets) > 0, "No datasets requested... nothing to do"
        self.phase1()
        self.phase2()


if __name__ == "__main__":
    #baseline = Baseline(verbose=True)
    baseline = Baseline(verbose=True, only_on_datasets=[utils.cifar10])
    baseline.go()
