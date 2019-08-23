from kardashigans.analyze_model import AnalyzeModel
from kardashigans.baseline import Baseline
from kardashigans.experiment import Experiment, ExperimentWithCheckpoints
from kardashigans.trainer import BaseTrainer
from keras.datasets import mnist, cifar10
import math
import numpy as np


class WinneryIntersection(ExperimentWithCheckpoints):
    """
    Experiment designed to check correlation between robustness of layer i
    of some trained model, and the number of non-zero weights in layer i of
    the winnery lotter ticket.
    """
    def __init__(self, prune_threshold=0.5, *args, **kwargs):
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
                                                  period=[],
                                                  *args, **kwargs)

    @staticmethod
    def get_model_name(dataset_name):
        return Baseline.get_model_name(dataset_name)

    @staticmethod
    def construct_dataset_trainer(dataset):
        # Train without pruning built-in, to compare pre-pruned model
        return Baseline.construct_dataset_trainer(dataset)

    def _get_robustness_list(self, trained_model, untrained_model, trainer, test_data):
        return [AnalyzeModel.calc_robustness(test_data=test_data,
                                             model=trained_model,
                                             source_weights_model=untrained_model,
                                             layer_indices=[i],
                                             batch_size=32)
                for i in trainer.get_weighted_layers_indices()]

    def _get_winnery_intersection_size_and_ratio(self, model, trainer):
        winnery_intersection_size = []
        winnery_intersection_ratio = []
        for i in trainer.get_weighted_layers_indices():
            # The winning ticket is the edge set consisting of non-zero weighted edges
            intersection_size = AnalyzeModel.total_edges_in_layers(model, [i], count_nonzero_only=True)
            total_weights = AnalyzeModel.total_edges_in_layers(model, [i])
            winnery_intersection_size.append(intersection_size)
            winnery_intersection_ratio.append(float(intersection_size) / float(total_weights))
        return winnery_intersection_size, winnery_intersection_ratio

    def go(self):
        unpruned_robustness = {}
        pruned_robustness = {}
        winnery_intersection_size = {}
        winnery_intersection_ratio = {}
        l2_norm_diff_map = {}
        l1_norm_diff_map = {}
        linf_norm_diff_map = {}
        untrained_prune_percent = {}
        trained_prune_percent = {}
        for model_name, trainer in self.get_trainer_map().items():
            test_data = self.get_test_data(model_name)
            layer_indices = trainer.get_weighted_layers_indices()
            glorot_indices = trainer.glorot_layer_indices()
            with self.open_model(model_name) as trained_model:
                with self.open_model_at_epoch(model_name, 'start') as untrained_model:
                    l2_norm_diff_map[model_name] = [AnalyzeModel.l2_diff(trained_model, untrained_model, i)
                                                    for i in layer_indices]
                    l1_norm_diff_map[model_name] = [AnalyzeModel.l1_diff(trained_model, untrained_model, i)
                                                    for i in layer_indices]
                    linf_norm_diff_map[model_name] = [AnalyzeModel.linf_diff(trained_model, untrained_model, i)
                                                      for i in layer_indices]
                    unpruned_robustness[model_name] = self._get_robustness_list(trained_model,
                                                                                untrained_model,
                                                                                trainer,
                                                                                test_data)
                    # Prune the trained model (before winnery intersection calculations)
                    BaseTrainer.prune_model(trained_model,
                                            self._prune_threshold,
                                            layer_indices,
                                            glorot_indices)
                    pruned_robustness[model_name] = self._get_robustness_list(trained_model,
                                                                              untrained_model,
                                                                              trainer,
                                                                              test_data)
                    trained_prune_percent[model_name] = \
                        AnalyzeModel.get_pruned_percent(trained_model, trainer.get_weighted_layers_indices())
                    winnery_intersection_size[model_name], winnery_intersection_ratio[model_name] = \
                        self._get_winnery_intersection_size_and_ratio(trained_model, trainer)
                    # For untrained prune percent we need to prune the untrained model
                    BaseTrainer.prune_model(untrained_model,
                                            self._prune_threshold,
                                            layer_indices,
                                            glorot_indices)
                    untrained_prune_percent[model_name] = \
                        AnalyzeModel.get_pruned_percent(untrained_model, trainer.get_weighted_layers_indices())
        for model_name in self.get_trainer_map().keys():
            AnalyzeModel.generate_winnery_graph(pruned_robustness=pruned_robustness[model_name],
                                                unpruned_robustness=unpruned_robustness[model_name],
                                                winnery_intersection_ratio=winnery_intersection_ratio[model_name],
                                                l2_diffs=l2_norm_diff_map[model_name],
                                                l1_diffs=l1_norm_diff_map[model_name],
                                                linf_diffs=linf_norm_diff_map[model_name],
                                                graph_name=model_name,
                                                output_dir=self._output_dir,
                                                filename=model_name + "_robustness_winnery_correlation")
            AnalyzeModel.generate_pruned_percent_graph(untrained_pruned_percent=untrained_prune_percent[model_name],
                                                       trained_pruned_percent=trained_prune_percent[model_name],
                                                       graph_name=model_name,
                                                       output_dir=self._output_dir,
                                                       filename=model_name + "_pruned_percent")
