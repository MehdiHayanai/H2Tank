from src.models.tank import Tank, Layer
from src.models.material import Material
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


class Individual:
    def __init__(self, genome: Genome):
        self.genome = genome  # list of float values

        self.tank: Tank = Tank(genome.get_genome(), internal_radius=174, length=1000)
        self.fitness = 0.0  # fitness value of the individual
        # ...

    def calculate_fitness(self):
        # Calculate fitness value of the individual
        self.fitness = ...
