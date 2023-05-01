from src.models.material import Material
import pandas as pd
import uuid


class Tank:
    """
    Represents a tank consisting of multiple layers.
    """

    def __init__(
        self,
        layers: list,
        internal_radius: float,
        length: float,
        pressure: float = 70.0,
        burst_test_pressure: float = 175.0,
    ):
        """
        Initializes a Tank object.

        Args:
        layers: a list of Layer objects representing the layers of the tank.
        internal_radius: the internal radius of the tank (in mm).
        length: the length of the tank (in mm).
        pressure: the pressure at which the tank operates (in MPa).
        burst_test_pressure: the pressure at which the tank is tested for burst (in MPa).
        """
        self.layers = layers
        self.internal_radius = internal_radius
        self.length = length
        self.pressure = pressure
        self.burst_test_pressure = burst_test_pressure
        self.number_of_layers = len(layers)

    def get_total_thickness(self):
        """
        Returns the total thickness of the tank.
        """
        return sum(layer.thickness for layer in self.layers)

    def get_external_radius(self):
        """
        Returns the external radius of the tank.
        """
        return self.internal_radius + self.get_total_thickness()

    def make_tank_excel(self, path: str = None):
        """
        Saves the layer characteristics of the tank as an Excel file.

        Args:
        path: the path where the Excel file should be saved. must end with /
        """
        data = [layer.get_layer_caracteristics() for layer in self.layers]

        data_frame = pd.DataFrame(data, columns=self.layers[0].columns)
        data_frame = data_frame[["angle", "thickness", "material_name", "layer_name"]]
        unique_filename = "Tank_data" + str(uuid.uuid4()) + ".xlsx"
        if path != None:
            unique_filename = path + unique_filename
        data_frame.to_excel(unique_filename)


class Layer:
    """
    Represents a single layer of the tank.
    """

    def __init__(
        self, thickness: float, angle: float, material: Material, name: str = "Unnamed"
    ):
        """
        Initializes a Layer object.

        Args:
        thickness: the thickness of the layer (in mm).
        angle: the angle of the layer (in degrees).
        material: a Material object representing the material of the layer.
        name: the name of the layer.
        """
        self.thickness = thickness
        self.material = material
        self.angle = angle
        self.name = name
        self.columns = ["thickness", "angle", "material_name", "layer_name"]

    def set_material(self, material):
        """
        Sets the material of the layer.

        Args:
        material: a Material object representing the new material of the layer.
        """
        self.material = material

    def get_inverse_layer(self):
        """
        Returns a layer with the same properties and inverse angle.
        """
        revers_layer = Layer(self.thickness, -self.angle, self.material, self.name)

        return revers_layer

    def get_layer_caracteristics(self):
        """
        Returns the layer characteristics as a tuple.
        """
        layer_caracteristics = (
            self.thickness,
            self.angle,
            self.material.name,
            self.name,
        )

        return layer_caracteristics

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        """
        return f"Thickness = {self.thickness} mm, angle = {self.angle}°, material {self.material.name}\n"
