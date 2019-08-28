from kardashigans.experiment import ExperimentWithCheckpoints
from kardashigans.analyze_model import AnalyzeModel
from kardashigans.resource_manager import ResourceManager


class SimpleExperiment(ExperimentWithCheckpoints):
    def __init__(self, epochs=100, batch_size=32, *args, **kwargs):
        """
        Abstract base class for layer-robustness experiments.

        For each trainer supplied, this experiment outputs a heatmap of
        robustness results.

        Columns represent the method used to define the layer weights
        (usually it's resetting to a specific epoch), and the rows are
        the different subsets of layers the method was applied to.

        :param epochs: Number of epochs each trainer trains. Note that
            we currently require this to be the same value for all
            trainers (simplifies the process of generating the heatmap
            columns) but there is no real reason we should limit this
            class like that FIXME
        :param batch_size: Like with epochs, this is a global value for
            ease of usage with get_full_robustness_results
        :param args: Remember to supply the Experiment class's required
            arguments!
        """
        self._epochs = epochs
        self._batch_size = batch_size
        super(SimpleExperiment, self).__init__(*args,
                                               period=self.get_period(),
                                               **kwargs)

    def get_layer_indices_list(self, trainer):
        """
        The actual rows of the output heatmap are controlled by the
        implementing class, which should also be aware of the number
        of parameter layers each trainer uses. For example, if we'd
        like to output two rows, the first one with robustness when
        we reset the first half of the layers and the second row with
        robustness when we reset the second half of the layers, then
        in the implementing class we do the following:

        all_layers = get_weighted_layers_indices()
        halfway = all_layers[0] + len(all_layers) // 2
        return [all_layers[:halfway], all_layers[halfway:]]
        """
        raise NotImplementedError

    def get_period(self):
        return sorted(list(set([(x * self._epochs) // 100 for x in [0, 1, 2, 3, 8, 40, 90]])))

    def go(self):
        for model_name, trainer in self.get_trainer_map().items():
            checkpoint_epochs = ResourceManager.get_checkpoint_epoch_keys(self.get_period())
            results, clean = self.get_full_robustness_results(model_name=model_name,
                                                              checkpoint_epochs=checkpoint_epochs,
                                                              layer_indices_list=self.get_layer_indices_list(trainer),
                                                              batch_size=self._batch_size)
            AnalyzeModel.generate_heatmap_from_results(heatmap_name=model_name,
                                                       results=results,
                                                       save_results_path=self._output_dir)
