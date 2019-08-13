import numpy as np
import collections
import matplotlib.pylab as plt
import matplotlib.pyplot as pyplot
import seaborn as sns
import csv
import os

from keras import backend as K
from kardashigans.verbose import Verbose
from keras import Model


class AnalyzeModel(object):
    """
    Static class for model analysis.
    """

    @staticmethod
    def l2_diff(model1, model2, layer, normalize=True):
        # get_weights()[0] is the edge weights, get_weights()[1] is the node biases.
        weights1 = model1.layers[layer].get_weights()[0]
        weights2 = model2.layers[layer].get_weights()[0]
        if normalize:
            weights1 = weights1 / np.linalg.norm(weights1)
            weights2 = weights2 / np.linalg.norm(weights2)
        return np.linalg.norm(weights1 - weights2)

    @staticmethod
    def _rernd_layers(model, layers_indices):
        session = K.get_session()
        for idx in layers_indices:
            layer = model.layers[idx]
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, 'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)

    @staticmethod
    def calc_robustness(test_data, model, source_weights_model=None, layer_indices=None, batch_size=32):
        """
        Evaluates the model on test data after re-initializing the layers
        with indices specified.

        Alternatively, if source_weights_model is given, sets the weights
        of the model (for layer indices appearing in layer_indices) at
        each given layer to the weights of source_weights_model.

        Function resets model weights to previous state before returning.

        :param test_data: A tuple (x_test, y_test) for validation
        :param model: The model to evaluate
        :param source_weights_model: The model from which we should
            copy weights and update our model's weights before eval.
            also accepts list of layer weights.
        :param layer_indices: Layers to reset the weights of.
        :param batch_size: used in evaluation
        :return: A number in the interval [0,1] representing accuracy.
        """
        if not layer_indices:
            layer_indices = []
        x_test, y_test = test_data
        prev_weights = model.get_weights()
        if source_weights_model:
            for idx in layer_indices:
                if isinstance(source_weights_model, Model):
                    loaded_weights = source_weights_model.layers[idx].get_weights()
                else:
                    loaded_weights = source_weights_model[idx]
                model.layers[idx].set_weights(loaded_weights)
        else:
            AnalyzeModel._rernd_layers(model, layer_indices)
        evaluated_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)
        model.set_weights(prev_weights)
        return evaluated_metrics[model.metrics_names.index('acc')]

    @staticmethod
    def get_weight_distances(model, source_weights_model, layer_indices=[], norm_orders=[]):
        """
        Computes distances between the layers of the given model and source model, in the chosen layers.
        Returns a dictionary in format: {idx: [dists (in the same order as the given list of distances)]}.
        """
        distance_list = collections.defaultdict(list)
        for layer in layer_indices:
            source_weights = source_weights_model.layers[layer].get_weights()
            model_weights = model.layers[layer].get_weights()
            if source_weights and model_weights:
                source_flatten_weights = np.concatenate([source_w.flatten() for source_w in source_weights])
                model_flatten_weights = np.concatenate([model_w.flatten() for model_w in model_weights])
                for order in norm_orders:
                    distance_list[layer].append(
                        np.linalg.norm(model_flatten_weights - source_flatten_weights, ord=order))
        return distance_list

    @staticmethod
    def generate_heatmap(data, row_labels, col_labels, filename, output_dir):
        """
        Creates a heatmap image from the data, outputs to file.

        :param data: List of lists of float values, indexed by data[row][column].
        :param row_labels: Size len(data) list of strings
        :param col_labels: Size len(data[0]) list of strings
        :param filename: Output filename, relative to the experiment results
            directory.
        """
        v = Verbose(name="AnalyzeModel.generate_heatmap")
        v.logger.debug("Generating heatmap. Data: {}".format(data))
        v.logger.debug("Rows: {}".format(row_labels))
        v.logger.debug("Cols: {}".format(col_labels))
        ax = sns.heatmap(data, linewidth=0.5, xticklabels=col_labels, yticklabels=row_labels)
        fig = ax.get_figure()
        fig.savefig(output_dir + filename)
        plt.show()

    @staticmethod
    def export_results_to_csv(results, row_labels, col_labels, filename, output_dir):
        head_row = [""] + col_labels
        with open(output_dir + filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(head_row)
            for exp_data, exp_name in zip(results.values(), row_labels):
                row = [exp_name] + list(exp_data.values())
                writer.writerow(row)

    @staticmethod
    def generate_heatmap_from_results(heatmap_name: str, results: dict, save_results_path: str):
        data = [list(value.values()) for value in results.values()]
        row_labels = ["Layers {}".format(layers) for layers in results.keys()]
        some_row = list(results.values())[0]
        col_labels = ["Epoch {}".format(e) for e in some_row.keys()]
        AnalyzeModel.generate_heatmap(data=data,
                                      row_labels=row_labels,
                                      col_labels=col_labels,
                                      filename="{}_heatmap.png".format(heatmap_name),
                                      output_dir=save_results_path)
        AnalyzeModel.export_results_to_csv(results=results, row_labels=row_labels, col_labels=col_labels,
                                           filename="{}_results.csv".format(heatmap_name),
                                           output_dir=save_results_path)

    @staticmethod
    def generate_winnery_graph(pruned_robustness,
                               unpruned_robustness,
                               winnery_intersection_ratio,
                               l2_diffs,
                               graph_name,
                               output_dir,
                               filename):
        pyplot.style.use('seaborn-darkgrid')
        pyplot.figure()
        palette = pyplot.get_cmap('Set1')
        pyplot.plot(pruned_robustness, marker='', color=palette(0), linewidth=1, alpha=0.9, label='Robustness (pruned)')
        pyplot.plot(unpruned_robustness, marker='', color=palette(1), linewidth=1, alpha=0.9, label='Robustness (unpruned)')
        pyplot.plot(winnery_intersection_ratio, marker='', color=palette(2), linewidth=1, alpha=0.9, label='Winning Ticket Intersection (ratio)')
        pyplot.plot(l2_diffs, marker='', color=palette(3), linewidth=1, alpha=0.9, label='L2 norm of weight difference')
        pyplot.legend()
        pyplot.title("Robustness & Winning Ticket Intersection by Layer\n(output to {})".format(graph_name),
                     loc='right', fontsize=12, fontweight=0, color='orange')
        pyplot.xlabel("Layer")
        pyplot.savefig(os.path.join(output_dir, filename), format='png')
