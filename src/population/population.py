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
        """
        Initializes a Population object.

        Args:
            population_size (int): The size of the population.
            materials (list[Material]): A list of Material objects representing the available materials.
            ref_angles (list[float]): A list of reference angles.
            burst_pressure (float, optional): The burst pressure threshold. Defaults to 158.
            fail_tol (float, optional): The failure tolerance. Defaults to 0.3.
            marge (float, optional): The margin value. Defaults to 1.
            coefs (tuple[float], optional): A tuple of coefficients. Defaults to (1.0, 0.5, -0.4).
            previous_population (list[Individual], optional): A list of Individual objects representing the previous population. Defaults to [].
        """
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
        """
        Initializes the population by creating the initial individuals.

        Args:
            starting_layers (int, optional): The number of starting layers. Defaults to 2.
        """

        # Create initial population of individuals

        offset = len(self.individuals)

        for i in range(self.population_size - offset):
            # Generate random values for parameters
            random_int = np.random.randint(low=5, high=10)
            random_number_of_layers = np.random.randint(low=0, high=50)

            # Generate random reference thicknesses
            ref_thickness = (np.random.rand(6) * random_int).tolist()

            # Define the number of material layers
            number_of_mat_layers = [starting_layers, random_number_of_layers]

            # Create a genome object for the individual
            genome = Genome(
                self.materials, number_of_mat_layers, self.ref_angles, ref_thickness
            )

            # Create a new individual with the genome and specified parameters
            self.individuals.append(Individual(genome, 5, self.fail_tol))

            # Set the burst test pressure for the individual's tank
            self.individuals[i + offset].tank.burst_test_pressure = self.burst_pressure

    def calculate_population_fitness(self):
        """
        Calculates the fitness for each individual in the population.

        This method iterates through each individual in the population and calculates their fitness using the provided coefficients and margin value.

        Returns:
            None
        """

        for individual in self.individuals:
            individual.calculate_fitness(self.coefs, self.marge)

    def select_individuals(self, top_idiv_prop=0.3):
        """
        Selects top individuals from the population.

        This method uses a selection mechanism to determine the top individuals based on their fitness scores. It sorts the individuals based on their fitness in descending order and selects a proportion of the top individuals specified by the 'top_idiv_prop' parameter.

        Args:
            top_idiv_prop (float, optional): Proportion of top individuals to select. Defaults to 0.3.

        Returns:
            list[Individual]: A list of top individuals selected from the population.
        """

        # Select individuals using some selection mechanism
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        top_proportion = int(len(self.individuals) * top_idiv_prop)
        top_individuals = self.individuals[:top_proportion]
        return top_individuals

    def reproduce(self, selected_individuals: list[Individual]):
        """
        Reproduces selected individuals to generate offspring.

        This method takes a list of selected individuals and generates offspring by performing reproduction between pairs of individuals. Each selected individual will reproduce with every other selected individual in a pairwise manner. The offspring generated from each pair is collected into a list and returned.

        Args:
            selected_individuals (list[Individual]): A list of selected individuals for reproduction.

        Returns:
            list[Individual]: A list of offspring generated from the selected individuals.
        """

        offspring: list[Individual] = []

        for indx, individual in enumerate(selected_individuals):
            for second_individual in selected_individuals[indx:]:
                offspring = (
                    offspring
                    + individual.reproduce(second_individual)
                    + second_individual.reproduce(individual)
                )

        return offspring

    def evolve(self):
        """
        Performs one iteration of the evolution process.

        This method represents one iteration of the evolution process. It selects individuals, produces offspring through reproduction, evaluates their fitness, and creates a new population based on the offspring with higher fitness than the mean fitness of the selected individuals. The new population is then returned.

        Returns:
            Population: The next generation population.
        """

        # Perform one iteration of evolution
        selected_individuals = self.select_individuals()
        offspring = self.reproduce(selected_individuals)
        print("Individuals", len(self.individuals))
        print("offspring", len(offspring))
        final_generation: list[Individual] = []

        # Calculate the mean fitness of the selected individuals
        mean_fitness_of_selected_individuals = np.array(
            [
                [
                    selected_individual.fitness
                    for selected_individual in selected_individuals
                ]
            ]
        ).mean()

        # Evaluate fitness of offspring and add individuals with fitness greater than or equal to the mean
        for individual in offspring:
            individual.calculate_fitness(self.coefs, self.marge)
            if individual.fitness >= mean_fitness_of_selected_individuals:
                final_generation.append(individual)

        # Sort the final generation based on fitness in descending order
        final_generation.sort(key=lambda x: x.fitness, reverse=True)

        # Create the next population with the top individuals from the final generation
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
