from src.population.individual import Genome, Individual
from src.models.material import Material
from src.models.tank import Tank


ref_angles = [88.7, 17.5, 28.0, 43.5, 72.0, 12.8, 18.4, 35.0, 77.0, 12.1]
ref_thickness = [0.1, 0.41, 0.515]
# Define materials
mat1 = Material((135000, 8500, 8500), (4400, 3053, 4400), (0.34, 0.31, 0.34), name="1")
mat2 = Material((40510, 12000, 12000), (3500, 2800, 3500), (0.22, 0.3, 0.22), name="2")
materials = [mat1, mat2]
number_of_mat_layers = [2, 78]

genome = Genome(materials, number_of_mat_layers, ref_angles, ref_thickness)

individual = Individual(genome)

tank: Tank = individual.tank

print(tank.layers[:2])
tank.make_tank_excel("gens/")
