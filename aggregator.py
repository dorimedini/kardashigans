from kardashigans.verbose import Verbose
import os


class Aggregator(Verbose):
    def __init__(self,
                 results_paths_list,
                 single_results_callback=None,
                 aggregate_results_callback=None):
        """
        :param results_paths_list: See generate_paths_from_timespan for an example
        :param single_results_callback: (str) -> (object). Can replace compute_single_result if you don't want to
            inherit and implement a new class for a specific aggregator.
        :param aggregate_results_callback: (dict[str:object]) -> (object). Like single_results_callback, designed to
            replace aggregate_results.
        """
        super(Aggregator, self).__init__("Aggregator")
        self._paths = results_paths_list
        self._single_call = single_results_callback
        self._aggregate_call = aggregate_results_callback

    @staticmethod
    def generate_paths_from_timespan(experiment_dir_name,
                                     s_hour, s_min, s_sec,
                                     t_hour, t_min, t_sec,
                                     day, month, year=2019):
        """
        Constructs a list of fully qualified paths to results directories given the directory name of the experiment.
        The list constructed may point to any number of directories with the SAME DAY in the given time range.
        For example, if kardashigans/WinneryIntersection directory has:
        kardashigans/
            ...
            WinneryIntersection/
                ...
                31_08_2019___20_17_32
                31_08_2019___19_42_53
                31_08_2019___17_50_56
                31_08_2019___14_20_12
                31_08_2019___14_15_18
                31_08_2019___13_59_21
                31_08_2019___13_50_13
                ...
        and you want to aggregate results computed between 13:55 and 19:45, call the method with:
          experiment_dir_name="WinneryIntersection",
          s_hour=13, s_min=55, s_sec=0,
          t_hour=19, t_min=45, t_sec=0,
          day=31, month=8

        :param experiment_dir: e.g. "Baseline", or "WinneryIntersection"
        :param s_hour/min/sec: Source time (inclusive)
        :param t_hour/min/sec: Target time (inclusive)
        :param day/month/year: All directories must be computed on this day (because I'm lazy)
        :return: A list of paths
        """
        source_dir = os.path.dirname(os.path.abspath(__file__))
        kardashigans_dir = os.path.abspath(os.path.join(source_dir, os.pardir))
        experiment_results_path = os.path.join(kardashigans_dir, experiment_dir_name)

        def time_leq(h1, m1, s1, h2, m2, s2):
            return h1*60*60 + m1*60 + s1 <= h2*60*60 + m2*60 + s2

        def get_time_trio(time_string):
            return [int(x) for x in time_string.split("_")[-3:]]

        all_dirs = os.listdir(experiment_results_path)
        todays_dirs = [x for x in all_dirs if x.startswith("{:02d}_{:02d}_{:04d}".format(day, month, year))]
        return_dirs = [x for x in todays_dirs if time_leq(*get_time_trio(x), t_hour, t_min, t_sec)
                       and time_leq(s_hour, s_min, s_sec, *get_time_trio(x))]
        return [os.path.join(experiment_results_path, x) for x in return_dirs]

    def compute_single_result(self, result_path):
        """
        Given a path to one of the results directories, computes the data relevant to this
        specific model / model set / whatever.

        Should return an object which will be passed to the aggregate_results method

        :param result_path: A string, fully qualified path to directory in which the resources reside.
        :return: Any object
        """
        if self._single_call:
            return self._single_call(result_path)
        raise NotImplementedError

    def aggregate_results(self, results_list):
        """
        :param results_list: A dictionary keyed by paths containing the data computed in compute_single_result
        :return: The aggregated data
        """
        if self._aggregate_call:
            return self._aggregate_call(results_list)
        raise NotImplementedError

    def go(self):
        return self.aggregate_results({path: self.compute_single_result(path) for path in self._paths})
