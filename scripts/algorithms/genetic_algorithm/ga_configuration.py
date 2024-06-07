from solution.generator.solution_generators import *
from reproduction.reproduction_strategy import *
from typing import Dict

class BaseGeneticAlgorithmConfiguration(object):
    def __init__(self, initial_solution_generator: AbstractGeneticAlgorithmSolutionGenerator = None):
        raise NotImplementedError
        
    def get_initial_population_size(self) -> int:
        raise NotImplementedError
    
    def get_initial_solution_generator(self) -> AbstractGeneticAlgorithmSolutionGenerator:
        raise NotImplementedError
    
    def get_reproducion_strategy(self) -> AbstractReproducionStrategy:
        raise NotImplementedError
        
    def get_reproducion_configuration(self) -> Dict:
        raise NotImplementedError
    
    def get_cnt_epoches(self) -> int:
        raise NotImplementedError