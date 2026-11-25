from _base import BaseAnalyzeAndVisualization
from analizer import ResultsAnalizerParent
from utils.interfaces_utils import measure_runtime

__all__ = ["AnalyzeAndVisualizationFunctionsInterface"]


class AnalyzeAndVisualizationFunctionsInterface(BaseAnalyzeAndVisualization):
    def __init__(self, verbose=True, output_directory=None):
        super().__init__(verbose, output_directory)

    @measure_runtime("Analyze folds results")
    def analyze_and_visualize_folds_results_gradient_descent_version(self, folds_outputs_directories_names):
        self.create_required_analyze_results_directories()
        analyzer = ResultsAnalizerParent(self._analyzes_directory, self._visulizations_directory, self._logger)
        analyzer.analyze_folds_results_gradient_descent_version(folds_outputs_directories_names)
        self._logger.info("results analysis completed successfully")
        return self._analyzes_directory
