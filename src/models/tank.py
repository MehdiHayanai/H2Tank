from src.models.material import Material
import pandas as pd
import uuid


# This file contains the Tank class which represents a tank consisting of multiple layers
# The Tank class has properties for length, diameter, and a list of Layer objects representing the layers of the tank


class Tank:
    def __init__(
        self,
        layers: list,
        internal_radius: float,
        length: float,
        pressure: float = 70.0,
        burst_test_pressure: float = 175.0,
    ):
        self.layers = layers
        self.internal_radius = internal_radius  # mm
        self.length = length  # mm

        self.pressure = pressure  # Mpa
        self.burst_test_pressure = burst_test_pressure  # Mpa
        self.number_of_layers = len(layers)

    def get_total_thickness(self):
        return sum(layer.thickness for layer in self.layers)

    def get_external_diameter(self):
        return self.internal_radius + self.get_total_thickness()

    def make_tank_excel(self, path: str = None):
        data = [layer.get_layer_caracteristics() for layer in self.layers]

        data_frame = pd.DataFrame(data, columns=self.layers[0].columns)
        data_frame = data_frame[["angle", "thickness", "material_name", "layer_name"]]
        unique_filename = "Tank_data" + str(uuid.uuid4()) + ".xlsx"
        if path != None:
            unique_filename = path + unique_filename
        data_frame.to_excel(unique_filename)


# This class  contains the Layer class which represents a single layer of the tank
# The Layer class has properties for thickness, angle, and a Material object representing the material of the layer


class Layer:
    def __init__(
        self, thickness: float, angle: float, material: Material, name: str = "Unnamed"
    ):
        self.thickness = thickness  # mm
        self.material = material
        self.angle = angle  # degree
        self.name = name

        self.columns = ["thickness", "angle", "material_name", "layer_name"]

    def set_material(self, material):
        self.material = material

    def get_inverse_layer(self):
        # creates layer with same properties and inverse angle
        revers_layer = Layer(self.thickness, -self.angle, self.material, self.name)

        return revers_layer

    def get_layer_caracteristics(self):
        layer_caracteristics = (
            self.thickness,
            self.angle,
            self.material.name,
            self.name,
        )
        """
            returns thickness, angle, material name, layer name
        """
        return layer_caracteristics

    def __repr__(self) -> str:
        return f"Thickness = {self.thickness} mm, angle = {self.angle}°, material {self.material.name} :\n"  # material {self.material.name}
