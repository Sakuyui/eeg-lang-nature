from .experiment_configuration import *
class ExperimentExecutor(object):
    def __init__(self):
        pass
    def do_experiments(self, experiment_configurations, source_signal_data):
        for experiment_configuration in experiment_configurations:
            experiment_configuration.do_experiment(source_signal_data)
            
            
