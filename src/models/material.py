# This file contains the Material class which represents a material used in the tank's construction
# The Material class has properties for Young's modulus, shear moduli, and Poisson's ratio


class Material:
    def __init__(self, young_modulus, shear_modulus, poisson_ratio):
        self.young_modulus = young_modulus  # MPa
        self.shear_modulus = shear_modulus  # MPa
        self.poisson_ratio = poisson_ratio
