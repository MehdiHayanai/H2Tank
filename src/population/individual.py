from src.models.tank import Tank, Layer
from src.models.material import Material
from src.models.computation import Computation
import numpy as np
import uuid


class Genome:
    def __init__(
        self,
        materials: list[Material] = None,
        layers_by_materials: list = None,
        ref_angles: list[float] = None,
        angles_to_thickness: dict = None,
        reverse: bool = True,
        mixed: bool = False,  # TO DO if time is available
    ):
        self.materials = materials
        self.layers_by_materials = layers_by_materials
        self.ref_angles = ref_angles
        self.angles_to_thickness = angles_to_thickness
        self.reverse = reverse
        self.mixed = mixed
        self.layers = []

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
            random_thickness = self.angles_to_thickness[random_angle]

            # Create a layer object using the random angle, thickness, and material
            layer = Layer(random_thickness, random_angle, mat)

            # Add the layer to the list
            layers.append(layer)

            # If reverse is True, add the inverse layer as well
            if self.reverse:
                layers.append(layer.get_inverse_layer())

        # Return the list of layers
        self.layers = layers
        return layers

    def get_genome(self):
        if len(self.layers) != 0:
            return self.layers

        layers: list[Layer] = []
        for number_of_layers, material in zip(self.layers_by_materials, self.materials):
            mat_layer = self.make_layers(number_of_layers, material)

            layers = layers + mat_layer

        return layers

    def set_layers(self, layers):
        self.layers = layers


class Individual(Computation):
    def __init__(
        self,
        genome: Genome,
        number_of_points_by_layer: int,
        failure_tol: float,
        internal_radius: float = 174,
        length=1,
        id=-1,
    ):
        self.genome = genome  # list of float values
        self.id = id
        self.internal_radius: float = internal_radius
        self.length = length
        self.number_of_points_by_layer = number_of_points_by_layer
        self.tank: Tank = Tank(genome.get_genome(), internal_radius, length)
        self.failure_tol = failure_tol
        self.origin_operation = "init"
        self.angles_ratios = []

        super().__init__(self.tank, self.number_of_points_by_layer)
        self.fitness = 0.0  # fitness value of the individual

    def calculate_fitness(
        self,
        coefs: tuple,
        marge: float,
        angles_prop: tuple[float],
        angles_ranges: tuple[float],
    ):
        self.calculate_deformation_and_constraints()
        tsai_wu = self.tsai_wu_failure_criteria()
        self.angles_ratios = self.angles_prop(angles_ranges)
        n1, n2, n3 = coefs

        def calculate_ponderation(f):
            if f < 1:
                return n1
            elif 1 <= f < 1 + marge:
                return n2
            else:
                return n3

        def make_triangle_function(x_target):
            f = (
                lambda x: -x / (1 - x_target) + 1 / (1 - x_target)
                if x >= x_target
                else x / x_target
            )
            return f

        c1 = (np.vectorize(calculate_ponderation)(tsai_wu)).sum()

        failure_ratio_f = make_triangle_function(self.failure_tol)
        self.failed = tsai_wu[tsai_wu > 1].shape[0] / tsai_wu.shape[0]
        self.nomber_ratio = failure_ratio_f(self.failed)

        c2 = self.nomber_ratio

        c3 = 1
        for indx, prop in enumerate(angles_prop):
            f = make_triangle_function(prop)
            c3 *= f(self.angles_ratios[indx])
        self.angles_prop_ratio = c3

        # Criteria 4
        self.inverse_radius_ratio = (
            self.tank.internal_radius / self.tank.get_external_radius()
        )
        c4 = self.inverse_radius_ratio

        self.fitness = (c4**8) * (c3**4) * (c2**3) * c1**1

    def angles_prop(self, angles_ranges):
        props = []
        layers_angles = [layer.angle for layer in self.tank.layers]
        for i in range(1, len(angles_ranges)):
            prop = 0
            for angle in layers_angles:
                if angles_ranges[i - 1] <= np.abs(angle) < angles_ranges[i]:
                    prop += 1
            props.append(prop / len(layers_angles))

        return props

    def reproduce(self, second_parent):
        first_parent_layers: list[Layer] = self.tank.layers
        first_parent_layers_mid = int(len(first_parent_layers) // 2)
        second_parent_layers: list[Layer] = second_parent.tank.layers
        second_parent_layers_mid = int(len(second_parent_layers) // 2)
        all_parents_layers = list(first_parent_layers) + list(second_parent_layers)

        first_child_layers: list[Layer] = (
            first_parent_layers[:first_parent_layers_mid]
            + second_parent_layers[:second_parent_layers_mid]
        )
        second_child_layers: list[Layer] = (
            first_parent_layers[first_parent_layers_mid:]
            + second_parent_layers[:second_parent_layers_mid]
        )
        third_child_layers: list[Layer] = (
            first_parent_layers[first_parent_layers_mid:]
            + second_parent_layers[second_parent_layers_mid:]
        )

        # fourth_child_layers: list[Layer] = np.random.choice(
        #     all_parents_layers,
        #     first_parent_layers_mid + second_parent_layers_mid,
        #     replace=True,
        # ).tolist()

        children_layers = [
            first_child_layers,
            second_child_layers,
            third_child_layers,
            # fourth_child_layers,
        ]

        new_individuals: list[Individual] = []
        for child_layer in children_layers:
            genome = Genome()
            genome.set_layers(child_layer)
            tmp_indiv = Individual(
                genome,
                self.number_of_points_by_layer,
                self.failure_tol,
                self.internal_radius,
                self.length,
            )

            new_individuals.append(tmp_indiv)

        return new_individuals
