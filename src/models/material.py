# This file contains the Material class which represents a material used in the tank's construction
# The Material class has properties for Young's modulus, shear moduli, and Poisson's ratio


class Material:
    def __init__(
        self,
        young_modulus: tuple,
        shear_modulus: tuple,
        poisson_ratios: tuple,
        Xt: float,
        Xc: float,
        Yt: float,
        Yc: float,
        X12: float,
        name: str = "Unnamed",
    ):
        self.young_modulus = young_modulus  # MPa
        self.shear_modulus = shear_modulus  # MPa
        self.poisson_ratios = poisson_ratios
        # Xt is the tensile strength of the composite material along the fiber direction MPa
        # Xc is the compressive strength of the composite material along the fiber direction Mpa
        self.Xt = Xt
        self.Xc = Xc

        # Yt is the tensile strength of the composite material perpendicular to the fiber direction Mpa
        # Yc is the compressive strength of the composite material perpendicular to the fiber direction Mpa
        self.Yt = Yt
        self.Yc = Yc
        self.X12 = X12

        self.name = name

    def get_constants(self):
        return (self.Xt, self.Xc, self.Yt, self.Yc, self.X12)
