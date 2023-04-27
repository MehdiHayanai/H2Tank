import numpy as np
from src.models.tank import Tank, Layer
from src.models.material import Material


# Define reference angles and thicknesses
ref_angles = [88.7, 17.5, 28.0, 43.5, 72.0, 12.8, 18.4, 35.0, 77.0, 12.1]
ref_thickness = [0.1, 0.41, 0.515]

# Define materials
mat1 = Material((135000, 8500, 8500), (4400, 3053, 4400), (0.34, 0.31, 0.34), name="1")
mat2 = Material((40510, 12000, 12000), (3500, 2800, 3500), (0.22, 0.3, 0.22), name="2")

# Define the number of layers for each material
number_of_layers_mat1 = 2
number_of_layers_mat2 = 78


def make_layers(number_of_layer, mat, ref_angles, ref_thickness, reverse=True):
    """
    This function creates a list of layer objects using the provided material, reference angles, and thicknesses.
    If reverse is True, it creates both the layer and its inverse layer.
    """
    diviseur = int(reverse) + 1
    layers = []

    for _ in range(int(number_of_layer / diviseur)):
        # Choose a random angle and thickness from the reference lists
        random_angle = np.random.choice(ref_angles)
        random_thickness = np.random.choice(ref_thickness)

        # Create a layer object using the random angle, thickness, and material
        layer = Layer(random_thickness, random_angle, mat)

        # Add the layer to the list
        layers.append(layer)

        # If reverse is True, add the inverse layer as well
        if reverse:
            layers.append(layer.get_inverse_layer())

    # Return the list of layers
    return layers


# Create the layer lists for each material
layers_mat1 = make_layers(number_of_layers_mat1, mat1, ref_angles, ref_thickness)
layers_mat2 = make_layers(number_of_layers_mat2, mat2, ref_angles, ref_thickness)

layers = layers_mat1 + layers_mat2

tank = Tank(layers, 400, 1000)

print(layers[0].get_layer_caracteristics())

# Print the number of reference angles and the number of unique reference angles
print(layers_mat1)
print(layers_mat2[:2])
print("Thickness", tank.get_total_thickness())
tank.make_tank_excel(path="gens/")
