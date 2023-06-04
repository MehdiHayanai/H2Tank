from src.population.individual import Individual, Genome
from src.models.material import Material
from src.models.tank import Layer
import numpy as np

# from src.populatio
"""
    Must edit after
"""


class Population:
    angles_to_thickness = {
        13.5: 1.03,
        13.2: 1.03,
        13: 1.03,
        17.5: 1.03,
        28: 1.03,
        43.5: 1.03,
        72: 1.03,
        12.8: 1.03,
        18.4: 1.03,
        35: 1.03,
        77: 1.03,
        12.1: 1.03,
        19.5: 1.03,
        48: 1.03,
        88.7: 2.46,
        88.8: 2.46,
    }

    def __init__(
        self,
        population_size: int,
        materials: list[Material],
        ref_angles: list[float],
        angles_to_thickness: dict = angles_to_thickness,
        burst_pressure: float = 175,
        fail_tol: float = 0.3,
        internal_radius: float = 174,  # mm
        marge: float = 0.2,
        coefs: tuple[float] = (1.0, 0.5, -0.4),
        previous_population=[],
        starting_layer: int = 0,
        number_of_layers_low: int = 5,
        number_of_layers_high: int = 50,
        number_of_points_by_layer: int = 5,
        angles_prop: tuple[float] = (1 / 3, 1 / 3, 1 / 3),
        angle_ranges: tuple[float] = (0, 30, 60, 90),
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
        self.angles_to_thickness: dict = angles_to_thickness
        self.materials: list[Material] = materials
        self.individuals: list[
            Individual
        ] = previous_population  # list of Individual objects
        self.population_fitness: list[float] = []
        self.starting_layer: int = starting_layer
        self.number_of_layers_low: int = number_of_layers_low
        self.number_of_layers_high: int = number_of_layers_high
        self.number_of_points_by_layer: int = number_of_points_by_layer
        self.angles_prop: tuple[float] = angles_prop  # (ξ1, ξ2, ξ3)
        self.angle_ranges: tuple[float] = angle_ranges  # (θ1, θ2, θ3, θ4)
        self.internal_radius: float = internal_radius

    def initialize(self):
        """
        Initializes the population by creating the initial individuals.

        Args:

        """

        # Create initial population of individuals

        offset = len(self.individuals)

        for i in range(self.population_size - offset):
            # Generate random values for parameters

            random_number_of_layers = np.random.randint(
                low=self.number_of_layers_low, high=self.number_of_layers_high
            )

            # Define the number of material layers
            number_of_mat_layers = [self.starting_layer, random_number_of_layers]

            # Create a genome object for the individual
            genome = Genome(
                self.materials,
                number_of_mat_layers,
                self.ref_angles,
                self.angles_to_thickness,
            )

            # Create a new individual with the genome and specified parameters
            self.individuals.append(
                Individual(
                    genome,
                    self.number_of_points_by_layer,
                    self.fail_tol,
                    internal_radius=self.internal_radius,
                )
            )

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
            individual.calculate_fitness(
                self.coefs, self.marge, self.angles_prop, self.angle_ranges
            )

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
            individual.calculate_fitness(
                self.coefs, self.marge, self.angles_prop, self.angle_ranges
            )
            if individual.fitness >= mean_fitness_of_selected_individuals:
                final_generation.append(individual)

        # Sort the final generation based on fitness in descending order
        final_generation.sort(key=lambda x: x.fitness, reverse=True)

        # Create the next population with the top individuals from the final generation

        next_population = Population(
            population_size=self.population_size,
            materials=self.materials,
            ref_angles=self.ref_angles,
            angles_to_thickness=self.angles_to_thickness,
            burst_pressure=self.burst_pressure,
            fail_tol=self.fail_tol,
            internal_radius=self.internal_radius,
            marge=self.marge,
            coefs=self.coefs,
            previous_population=final_generation[: self.population_size],
            starting_layer=self.starting_layer,
            number_of_layers_low=self.number_of_layers_low,
            number_of_layers_high=self.number_of_layers_high,
            number_of_points_by_layer=self.number_of_points_by_layer,
            angles_prop=self.angles_prop,
            angle_ranges=self.angle_ranges,
        )

        return next_population
