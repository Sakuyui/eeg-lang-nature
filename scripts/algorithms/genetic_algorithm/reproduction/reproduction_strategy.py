

class AbstractReproducionStrategy(object):
    def generate_next_generation(self, current_population, fitness, extra_configuration) -> Tuple[List[AbstractGASolution], List[float]]:
        raise NotImplementedError
    
    