from src.models.tank import Tank, Layer
from src.models.material import Material
from src.models.computation import Computation
import numpy as np


class Genome:
    def __init__(
        self,
        materials: list[Material],
        layers_by_materials: list,
        ref_angles: list[float],
        ref_thickness: list[float],
        reverse: bool = True,
        mixed: bool = False,  # TO DO if time is available
    ):
        self.materials = materials
        self.layers_by_materials = layers_by_materials
        self.ref_angles = ref_angles
        self.ref_thickness = ref_thickness
        self.reverse = reverse
        self.mixed = mixed

    def make_layers(self, number_of_layers, mat):
        """
        This function creates a list of layer objects using the provided material, reference angles, and thicknesses.
        If reverse is True, it creates both the layer and its inverse layer.
        """
        diviseur = int(self.reverse) + 1
        layers: list[Layer] = []

        for _ in range(int(number_of_layers / diviseur)):
            # Choose a random angle and thickness from the reference lists
            random_angle = np.random.choice(self.ref_angles)
            random_thickness = np.random.choice(self.ref_thickness)

            # Create a layer object using the random angle, thickness, and material
            layer = Layer(random_thickness, random_angle, mat)

            # Add the layer to the list
            layers.append(layer)

            # If reverse is True, add the inverse layer as well
            if self.reverse:
                layers.append(layer.get_inverse_layer())

        # Return the list of layers
        return layers

    def get_genome(self):
        layers: list[Layer] = []

        for number_of_layers, material in zip(self.layers_by_materials, self.materials):
            mat_layer = self.make_layers(number_of_layers, material)

            layers = layers + mat_layer

        return layers


class Individual(Computation):
    def __init__(
        self,
        genome: Genome,
        number_of_points_by_layer: int,
        failure_tol: float,
        internal_radius=174,
        length=1000,
    ):
        self.genome = genome  # list of float values
        self.internal_radius = internal_radius
        self.length = length
        self.number_of_points_by_layer = number_of_points_by_layer
        self.tank: Tank = Tank(genome.get_genome(), internal_radius, length)
        self.failure_tol = failure_tol

        super().__init__(self.tank, self.number_of_points_by_layer)
        self.fitness = 0.0  # fitness value of the individual

    def calculate_fitness(self, coefs: tuple, marge: float):
        self.calculate_deformation_and_constraints()
        tsai_wu = self.tsai_wu_failure_criteria()

        n1, n2, n3 = coefs

        def calculate_ponderation(f):
            if f <= 1:
                return n1
            elif 1 < f <= 1 + marge:
                return n2
            else:
                return n3

        self.inverse_radius_ratio = (
            self.tank.internal_radius / self.tank.get_external_radius()
        )

        failure_ratio_f = lambda x: -x / (1 - self.failure_tol) + 1 / (
            1 - self.failure_tol
        )

        self.failed = tsai_wu[tsai_wu > 1].shape[0] / tsai_wu.shape[0]

        self.nomber_ratio = failure_ratio_f(self.failed)

        fintess = (
            self.inverse_radius_ratio**4
            * self.nomber_ratio
            * (np.vectorize(failure_ratio_f)(tsai_wu)).sum()
        )

        self.fitness = fintess
