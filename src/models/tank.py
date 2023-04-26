from src.models.material import Material

# This file contains the Tank class which represents a tank consisting of multiple layers
# The Tank class has properties for length, diameter, and a list of Layer objects representing the layers of the tank


class Tank:
    def __init__(
        self,
        layers: list,
        internal_diameter: float,
        length: float,
        pressure: float = 70.0,
        burst_test_pressure: float = 175.0,
    ):
        self.layers = layers
        self.internal_diameter = internal_diameter  # mm
        self.length = length  # mm

        self.pressure = pressure  # Mpa
        self.burst_test_pressure = burst_test_pressure  # Mpa
        self.number_of_layers = len(layers)

    def get_total_thickness(self):
        return sum(layer.thickness for layer in self.layers)

    def get_external_diameter(self):
        return self.internal_diameter + self.get_total_thickness()


# This class  contains the Layer class which represents a single layer of the tank
# The Layer class has properties for thickness, angle, and a Material object representing the material of the layer


class Layer:
    def __init__(self, thickness: float, material: Material, angle: float):
        self.thickness = thickness  # mm
        self.material = material
        self.angle = angle  # degree

    def set_material(self, material):
        self.material = material
