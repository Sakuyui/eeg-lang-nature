from typing import List, Dict, Tuple
from ga_configuration import *
from solution.ga_solution import *
from solution.generator.solution_generators import *
    
class AbstractGeneticAlgorithm(object):
    def __init__(self, ga_configuration: BaseGeneticAlgorithmConfiguration):
        self.ga_configuration = ga_configuration
    
    def do_natural_selection(self, solutions: List[AbstractGASolution]) -> Tuple[AbstractGASolution, List[float]]:
        raise NotImplementedError
    
    def event_epoch_begin(self, context: Dict):
        pass
    
    def event_epoch_end(self, context: Dict):
        pass
    
    def event_algorithm_finish(self, context: Dict):
        pass
        
    def do_genetic_algorithm(self):
        # get required objects and parameters from the algorithm configuration.
        initial_population_size = self.ga_configuration.get_initial_population_size()
        population_maximum_size = self.ga_configuration.get_initial_population_size()
        cnt_epoches = self.ga_configuration.get_cnt_epoches()
        
        # generate intial solutions.
        solutions: List[AbstractGASolution] = self.ga_configuration.get_initial_solution_generator()\
            .generate_init_solutions(initial_population_size)
        
        # apply natural selection if the amount of solutions larger than maximum population size.
        if len(solutions) > population_maximum_size:
            solutions, fitness = self.do_natural_selection(solutions)
            
        for epoch_index in range(cnt_epoches):
            self.event_epoch_begin({
                'solutions': solutions,
                'fitness': fitness,
                'ga_configuration': self.ga_configuration,
                'epoch_index': epoch_index 
            })
            
            # reproduce to generate new individuals.
            reproducion_strategy = self.ga_configuration.get_reproducion_strategy()
            solutions, fitness = reproducion_strategy.\
                generate_next_generation(solutions, fitness, self.ga_configuration.get_reproducion_configuration())
            
            # apply natural selection if the amount of solutions larger than maximum population size.
            if len(solutions) > population_maximum_size:
                solutions, fitness = self.do_natural_selection(solutions)
                
            self.event_algorithm_finish({
                'solutions': solutions,
                'fitness': fitness,
                'ga_configuration': self.ga_configuration,
                'epoch_index': epoch_index 
            })
            
        self.event_algorithm_finish({
                'solutions': solutions,
                'fitness': fitness,
                'ga_configuration': self.ga_configuration
        })
        
    