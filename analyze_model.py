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
    def l1_diff(model1, model2, layer, normalize=True):
        return AnalyzeModel.get_weight_distances(model1,
                                                 model2,
                                                 layer_indices=[layer],
                                                 norm_orders=[1],
                                                 normalize=normalize)[layer][0]

    @staticmethod
    def l2_diff(model1, model2, layer, normalize=True):
        return AnalyzeModel.get_weight_distances(model1,
                                                 model2,
                                                 layer_indices=[layer],
                                                 norm_orders=[2],
                                                 normalize=normalize)[layer][0]

    @staticmethod
    def linf_diff(model1, model2, layer, normalize=True):
        return AnalyzeModel.get_weight_distances(model1,
                                                 model2,
                                                 layer_indices=[layer],
                                                 norm_orders=[np.inf],
                                                 normalize=normalize)[layer][0]

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
    def total_edges(model):
        return AnalyzeModel.total_edges_in_layers(model, list(range(len(model.layers))))

    @staticmethod
    def total_edges_in_layers(model, layers=[], count_nonzero_only=False):
        edges = 0
        for layer in layers:
            weights = model.layers[layer].get_weights()
            if weights:
                if count_nonzero_only:
                    edges += np.count_nonzero(weights[0])
                else:
                    edges += weights[0].size
        return edges

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
    def get_weight_distances(model, source_weights_model, layer_indices=[], norm_orders=[], normalize=True):
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
                    weights1 = source_flatten_weights / np.linalg.norm(source_flatten_weights, ord=order) \
                        if normalize else source_flatten_weights
                    weights2 = model_flatten_weights / np.linalg.norm(model_flatten_weights, ord=order) \
                        if normalize else model_flatten_weights
                    distance_list[layer].append(
                        np.linalg.norm(weights1 - weights2, ord=order))
        return distance_list

    @staticmethod
    def generate_heatmap(data, row_labels, col_labels, filename, output_dir, graph_name=None):
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
        ax = sns.heatmap(data,
                         linewidth=0.5,
                         xticklabels=col_labels,
                         yticklabels=row_labels,
                         vmin=0,
                         vmax=1,
                         cmap='afmhot')
        ax.set_title(graph_name if graph_name else filename)
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
                               l1_diffs,
                               linf_diffs,
                               graph_name,
                               output_dir,
                               filename):
        pyplot.style.use('seaborn-darkgrid')
        pyplot.figure()
        palette = pyplot.get_cmap('Set1')
        column_list = list(range(len(unpruned_robustness)))
        pyplot.scatter(column_list, pruned_robustness, marker='o', color=palette(0), linewidth=1, alpha=0.9,
                       label='Robustness (pruned)')
        pyplot.scatter(column_list, unpruned_robustness, marker='o', color=palette(1), linewidth=1, alpha=0.9,
                       label='Robustness (unpruned)')
        pyplot.scatter(column_list, winnery_intersection_ratio, marker='o', color=palette(2), linewidth=1, alpha=0.9,
                       label='Winning Ticket Intersection (ratio)')
        pyplot.scatter(column_list, l2_diffs, marker='o', color=palette(3), linewidth=1, alpha=0.9,
                       label='L2 norm of weight difference')
        pyplot.scatter(column_list, l1_diffs, marker='o', color=palette(4), linewidth=1, alpha=0.9,
                       label='L1 norm of weight difference')
        pyplot.scatter(column_list, linf_diffs, marker='o', color=palette(5), linewidth=1, alpha=0.9,
                       label='Linf norm of weight difference')
        pyplot.legend(loc='center left', bbox_to_anchor=(-0.7, 0.5))
        pyplot.title("Robustness & Winning Ticket Intersection by Layer\n(output to {})".format(graph_name),
                     loc='right', fontsize=12, fontweight=0, color='orange')
        pyplot.xlabel("Layer")
        pyplot.savefig(os.path.join(output_dir, filename), format='png')

    @staticmethod
    def get_pruned_percent(model, layer_list=[]):
        pruned_percents = []
        for layer in layer_list:
            total = AnalyzeModel.total_edges_in_layers(model, [layer])
            pruned = total - AnalyzeModel.total_edges_in_layers(model, [layer], count_nonzero_only=True)
            pruned_percents.append(pruned / total)
        return pruned_percents

    @staticmethod
    def generate_transfer_graph(results: dict, output_dir: str, filename="transfer_fig", base_name="base"):
        v = Verbose(name="AnalyzeModel.generate_transfer_graph")
        for key, val in results.item:
            v.logger.debug("Layer: {}, results {}".format(key, val))
        pyplot.style.use('seaborn-darkgrid')
        palette = pyplot.get_cmap('Set1')
        base_results = results.pop(base_name)
        ab = [base_results["b"]]
        abp = [base_results["b"]]
        ba = [base_results["a"]]
        bap = [base_results["a"]]
        for result in results.values():
            ab.append(result["ab"])
            abp.append(result["abp"])
            ba.append(result["ba"])
            bap.append(result["bap"])
        x = [base_name] + list(results.keys())
        pyplot.figure()
        pyplot.scatter(x, ab, marker='*', color=palette(0), linewidth=1, alpha=0.9,
                    label='ab')
        pyplot.scatter(x, abp, marker='+', color=palette(0), linewidth=1, alpha=0.9,
                    label='ab+')
        pyplot.scatter(x, ba, marker='*', color=palette(1), linewidth=1, alpha=0.9,
                    label='ba')
        pyplot.scatter(x, bap, marker='+', color=palette(1), linewidth=1, alpha=0.9,
                    label='ba+')
        pyplot.legend()
        pyplot.title("Transfer strength by Layers copied", loc='right', fontsize=12, fontweight=0,
                     color='orange')
        pyplot.ylabel("Accuracy")
        pyplot.xlabel("Layers copied")
        pyplot.savefig(os.path.join(output_dir, filename + ".png"), format='png')
