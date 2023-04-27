from src.models.tank import Tank, Layer
from src.models.material import Material


ref_angles = [88.7, 17.5, 28, 43.5, 72, 12.8, 18.4, 35, 77, 12.1]
ref_epaisseur = [0.1, 0.41, 0.515]

mat1 = Material((135000, 8500, 8500), (4400, 3053, 4400), (0.34, 0.31, 0.34))
mat2 = Material((40510, 12000, 12000), (3500, 2800, 3500), (0.22, 0.3, 0.22))

number_of_layers_mat1 = 2
number_of_layers_mat1 = 38

print(len(ref_angles), len(set(ref_angles)))
