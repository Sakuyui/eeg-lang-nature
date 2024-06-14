from typing import Tuple, List, Dict, Callable
from solution.ga_solution import *
import random, numpy

class AbstractReproducionStrategy(object):
    def generate_next_generation(self, population: List[AbstractGASolution], \
        fitness: List[float], extra_configuration: Dict) -> Tuple[List[AbstractGASolution], List[float]]:
        raise NotImplementedError
        
class RuleAndParameterIndepdentReproducionStrategy(AbstractReproducionStrategy):
    def __init__(self, individual_selection_strategy
                    :Callable[[List[AbstractGASolution], List[int]], Tuple[AbstractGASolution, AbstractGASolution]],
                rule_crossover_strategy: Callable[[AbstractGASolution, AbstractGASolution], Tuple[AbstractGASolution, AbstractGASolution]], 
                rule_mutation_strategy: Callable[[AbstractGASolution], AbstractGASolution], 
                parameter_crossover_strategy:  Callable[[AbstractGASolution, AbstractGASolution], Tuple[AbstractGASolution, AbstractGASolution]],
                parameter_mutation_strategy: Callable[[AbstractGASolution], AbstractGASolution]) -> None:
        
        self.individual_selection_strategy = individual_selection_strategy
        self.rule_crossover_strategy = rule_crossover_strategy
        self.rule_mutation_strategy = rule_mutation_strategy
        self.parameter_crossover_strategy = parameter_crossover_strategy
        self.parameter_mutation_strategy = parameter_mutation_strategy
        
    def generate_next_generation(self, population, fitness, extra_configuration) -> Tuple[List[AbstractGASolution], List[float]]:
        max_population_size = extra_configuration['max_population_size']
        while(len(population) < max_population_size):
            individual1_id, individual2_id = self.individual_selection_strategy(population, fitness)
            individual1 = population[individual1_id]
            individual2 = population[individual2_id]
            # parameter such as start symbol id, rule possibilities are evoluated separately with grammar.
            individual1, individual2 = self.evolute_parameters(self.evolute_rules(individual1, individual2))

    def evolute_rules(self, individual1: AbstractGASolution, individual2: AbstractGASolution):
        individual1, individual2 = self.rule_crossover_strategy(individual1, individual2)
        individual1 = self.rule_mutation_strategy(individual1)
        individual2 = self.rule_mutation_strategy(individual2)
        return individual1, individual2
    
    def evolute_parameters(self, individual1: AbstractGASolution, individual2: AbstractGASolution):
        individual1, individual2 = self.parameter_crossover_strategy(individual1, individual2)
        individual1 = self.parameter_mutation_strategy(individual1)
        individual2 = self.parameter_mutation_strategy(individual2)
        return individual1, individual2

def rule_crossover(individual1: BytearrayRepresentedGASolution, individual2: BytearrayRepresentedGASolution):
    ind1 = individual1.data
    ind2 = individual2.data
    ind1_length = len(ind1)
    ind2_length = len(ind2)
    crossover_length_random = numpy.random.randint(0, min(ind1_length, ind1_length))
    split_point_index1 = numpy.random.randint(0, min(0, ind1_length - crossover_length_random))
    split_point_index2 = numpy.random.randint(0, min(0, ind2_length - crossover_length_random))
    
    # swap
    ind1_swap_segment = ind1[split_point_index1: split_point_index1 + crossover_length_random] 
    ind1[split_point_index1: split_point_index1 + crossover_length_random] = ind2[split_point_index2: split_point_index2 + crossover_length_random]
    ind2[split_point_index2: split_point_index2 + crossover_length_random] = ind1_swap_segment
    
    # shift
    shift_length = numpy.random.randint(0, ind1_length) # TODO: important sampling
    shift_segment_beginning = numpy.random.randint(0, ind1_length - shift_length)
    segment = ind1[shift_segment_beginning, shift_segment_beginning + shift_length]
    ind2.append(segment)
    ind1[shift_segment_beginning: ind1_length - shift_segment_beginning] = \
        ind1[-ind1_length + shift_segment_beginning: ]
    
    return BytearrayRepresentedGASolution(ind1), BytearrayRepresentedGASolution(ind2)

def rule_mutation(individual1: BytearrayRepresentedGASolution):
    ind1 = individual1.data
    
    # TODO: How we ranwaqdomly change the solution to a solution which near it inside the solution space.
    
    return BytearrayRepresentedGASolution(ind1)

