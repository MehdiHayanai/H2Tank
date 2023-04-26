class Individual:
    def __init__(self, genome):
        self.genome = genome  # list of float values
        self.fitness = 0.0  # fitness value of the individual
        # ...

    def calculate_fitness(self):
        # Calculate fitness value of the individual
        self.fitness = ...
