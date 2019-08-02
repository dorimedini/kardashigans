from kardashigans.analyze_model import AnalyzeModel
from kardashigans.baseline import Baseline
from kardashigans.experiment import Experiment
from kardashigans.trainer import BaseTrainer
from keras.datasets import mnist, cifar10
import math
import numpy as np
import pandas as pd


class WinneryIntersection(Experiment):
    """
    Experiment designed to check correlation between robustness of layer i
    of some trained model, and the number of non-zero weights in layer i of
    the winnery lotter ticket.
    """
    def __init__(self, prune_threshold, *args, **kwargs):
        assert not math.isnan(prune_threshold) and prune_threshold >= 0, \
            "Must init WinneryIntersection with non-negative float value for pruning threshold"
        self._prune_threshold = prune_threshold
        mnist_name = WinneryIntersection.get_model_name(Experiment.get_dataset_name(mnist))
        cifar10_name = WinneryIntersection.get_model_name(Experiment.get_dataset_name(cifar10))
        super(WinneryIntersection, self).__init__(name='WinneryIntersection',
                                                  model_names=[mnist_name, cifar10_name],
                                                  trainers={
                                                      mnist_name: WinneryIntersection.construct_dataset_trainer(mnist),
                                                      cifar10_name: WinneryIntersection.construct_dataset_trainer(cifar10)
                                                  },
                                                  *args, **kwargs)

    @staticmethod
    def get_model_name(dataset_name):
        return Baseline.get_model_name(dataset_name)

    @staticmethod
    def construct_dataset_trainer(dataset):
        # Train without pruning built-in, to compare pre-pruned model
        return Baseline.construct_dataset_trainer(dataset)

    def _get_robustness_list(self, model, trainer, test_data):
        return [AnalyzeModel.calc_robustness(test_data=test_data,
                                             model=model,
                                             layer_indices=[i],
                                             batch_size=32)
                for i in trainer.get_weighted_layers_indices()]

    def _get_winnery_intersection_size_and_ratio(self, model, trainer):
        winnery_intersection_size = []
        winnery_intersection_ratio = []
        for i in trainer.get_weighted_layers_indices():
            weights = model.layers[i].get_weights()
            # The winning ticket is the edge set consisting of non-zero weighted edges
            intersection_size = len([w for w in weights if w != 0])
            winnery_intersection_size.append(intersection_size)
            winnery_intersection_ratio.append(float(intersection_size) / float(len(weights)))
        return winnery_intersection_size, winnery_intersection_ratio

    def go(self):
        unpruned_robustness = {}
        pruned_robustness = {}
        winnery_intersection_size = {}
        winnery_intersection_ratio = {}
        for model_name, trainer in self.get_trainer_map().items():
            test_data = self.get_test_data(model_name)
            with self.open_model(model_name) as model:
                unpruned_robustness[model_name] = self._get_robustness_list(model, trainer, test_data)
                BaseTrainer.prune_trained_model(model, self._prune_threshold)
                pruned_robustness[model_name] = self._get_robustness_list(model, trainer, test_data)
                winnery_intersection_size[model_name], winnery_intersection_ratio[model_name] = \
                    self._get_winnery_intersection_size_and_ratio(model, trainer)
        for model_name in self.get_trainer_map().keys():
            AnalyzeModel.generate_robustness_winnery_correlation_graph(pruned_robustness[model_name],
                                                                       unpruned_robustness[model_name],
                                                                       winnery_intersection_ratio[model_name],
                                                                       self._output_dir,
                                                                       model_name + "_robustness_winnery_correlation")
        # TODO: Show winnery_intersection_ratio alongside unpruned / pruned robustness