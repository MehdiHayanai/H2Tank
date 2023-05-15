from src.population.individual import Individual, Genome
from src.models.material import Material
from src.models.tank import Layer
import numpy as np

# from src.populatio
"""
    Must edit after
"""


class Population:
    def __init__(
        self,
        population_size: int,
        materials: list[Material],
        ref_angles: list[float],
        burst_pressure: float = 158,
        fail_tol: float = 0.3,
        marge: float = 1,
        coefs: tuple[float] = (1.0, 0.5, -0.4),
        previous_population=[],
    ):
        self.population_size: int = population_size
        self.burst_pressure: float = burst_pressure
        self.fail_tol: float = fail_tol
        self.coefs: tuple[float] = coefs
        self.marge = marge
        self.ref_angles: list[float] = ref_angles
        self.materials: list[Material] = materials
        self.individuals: list[
            Individual
        ] = previous_population  # list of Individual objects
        self.population_fitness: list[float] = []

    def initialize(self, starting_layers=2):
        # Create initial population of individuals

        offset = len(self.individuals)

        for i in range(self.population_size - offset):
            random_int = np.random.randint(low=5, high=10)
            random_number_of_layers = np.random.randint(low=0, high=50)

            ref_thickness = (np.random.rand(6) * random_int).tolist()
            # Define materials

            number_of_mat_layers = [starting_layers, random_number_of_layers]
            # computation.calculate_matrices()
            genome = Genome(
                self.materials, number_of_mat_layers, self.ref_angles, ref_thickness
            )

            self.individuals.append(Individual(genome, 5, self.fail_tol))
            # set
            self.individuals[i + offset].tank.burst_test_pressure = self.burst_pressure

    def calculate_population_fitness(self):
        for individual in self.individuals:
            individual.calculate_fitness(self.coefs, self.marge)

    def select_individuals(self, top_idiv_prop=0.3):
        # Select individuals using some selection mechanism
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        top_proportion = int(len(self.individuals) * top_idiv_prop)
        top_individuals = self.individuals[:top_proportion]
        return top_individuals

    def reproduce(self, selected_individuals: list[Individual]):
        offspring: list[Individual] = []

        for indx, individual in enumerate(selected_individuals):
            for second_individual in selected_individuals[indx:]:
                offspring = (
                    offspring
                    + individual.reproduce(second_individual)
                    + second_individual.reproduce(individual)
                )
        return offspring

    # def evaluate(
    #     self, offspring: list[Individual], selected_individuals: list[Individual]
    # ):  # -> Population:
    #     final_generation: list[Individual] = []
    #     mean_fitness_of_selected_individuals = np.array(
    #         [
    #             selected_individual.fitness
    #             for selected_individual in selected_individuals
    #         ]
    #     ).mean()
    #     print(mean_fitness_of_selected_individuals)

    #     for individual in offspring:
    #         individual.calculate_fitness(self.coefs, self.marge)
    #         if individual.fitness >= mean_fitness_of_selected_individuals:
    #             final_generation.append(individual)

    #     return

    def evolve(self):
        # Perform one iteration of evolution
        selected_individuals = self.select_individuals()
        offspring = self.reproduce(selected_individuals)
        print("Individuals", len(self.individuals))
        print("offspring", len(offspring))
        final_generation: list[Individual] = []
        mean_fitness_of_selected_individuals = np.array(
            [
                [
                    selected_individual.fitness
                    for selected_individual in selected_individuals
                ]
            ]
        ).mean()

        for individual in offspring:
            individual.calculate_fitness(self.coefs, self.marge)
            if individual.fitness >= mean_fitness_of_selected_individuals:
                final_generation.append(individual)

        final_generation.sort(key=lambda x: x.fitness, reverse=True)
        next_population = Population(
            self.population_size,
            self.materials,
            self.ref_angles,
            self.burst_pressure,
            self.fail_tol,
            self.marge,
            self.coefs,
            final_generation[: self.population_size],
        )

        return next_population
