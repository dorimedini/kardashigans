from kardashigans.verbose import Verbose


class Aggregator(Verbose):
    def __init__(self, results_paths_list):
        super(Aggregator, self).__init__("Aggregator")
        self._paths = results_paths_list

    def compute_single_result(self, result_path):
        """
        Given a path to one of the results directories, computes the data relevant to this
        specific model / model set / whatever.

        Should return an object which will be passed to the aggregate_results method

        :param result_path: A string, fully qualified path to directory in which the resources reside.
        :return: Any object
        """
        raise NotImplementedError

    def aggregate_results(self, results_list):
        """
        :param results_list: A dictionary keyed by paths containing the data computed in compute_single_result
        :return: The aggregated data
        """
        raise NotImplementedError

    def go(self):
        return self.aggregate_results({path: self.compute_single_result(path) for path in self._paths})
