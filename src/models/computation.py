from src.models.tank import Tank, Layer
import numpy as np


class Computation:
    def __init__(self, tank: Tank, number_of_points_by_layer: int):
        self.tank = tank
        self.number_of_points_by_layer = number_of_points_by_layer
        self.number_of_layers = len(self.tank.layers)
        self.r_cl: np.array = np.zeros(self.number_of_layers + 1)
        self.r_cl[0] = tank.internal_radius
        self.a1: np.array = np.zeros(self.number_of_layers)
        self.a2: np.array = np.zeros(self.number_of_layers)
        self.b: np.array = np.zeros(self.number_of_layers)

        self.d: np.array = np.array([])
        self.e: np.array = np.array([])
        self.a: np.array = np.array([])

        self.Som: list[np.ndarray] = []
        self.Com: list[np.ndarray] = []
        self.Tsig: list[np.ndarray] = []
        self.Teps: list[np.ndarray] = []
        self.Tsig_1: list[np.ndarray] = []
        self.Teps_1: list[np.ndarray] = []
        self.Cgm: list[np.ndarray] = []

    def calculate_cost(self):
        # calculate cost function based on material resistance model
        cost = ...
        return cost

    def __calculate_matrices(self):
        # Get the list of layers from the tank object
        layers: list[Layer] = self.tank.layers

        # Loop through each layer and calculate its stiffness and compliance matrices
        for layer in layers:
            # Get the material properties for the current layer
            E1, E2, E3 = layer.material.young_modulus
            G23, G13, G12 = layer.material.shear_modulus
            v23, v13, v12 = layer.material.poisson_ratios

            # Calculate the 6x6 stiffness matrix
            stiffness_matrix = np.zeros((6, 6))
            stiffness_matrix[0][0] = 1 / E1
            stiffness_matrix[0][1] = -v23 / E1
            stiffness_matrix[1][0] = stiffness_matrix[0][1]
            stiffness_matrix[0][2] = -v12 / E1
            stiffness_matrix[2][0] = stiffness_matrix[0][2]
            stiffness_matrix[1][1] = 1 / E2
            stiffness_matrix[1][2] = -v13 / E2
            stiffness_matrix[2][1] = stiffness_matrix[1][2]
            stiffness_matrix[2][2] = 1 / E3
            stiffness_matrix[3][3] = 1 / G13
            stiffness_matrix[4][4] = 1 / G12
            stiffness_matrix[5][5] = 1 / G23

            # Calculate the 6x6 compliance matrix
            compliance_matrix = np.linalg.inv(stiffness_matrix)

            # Append the stiffness and compliance matrices to the Som and Com lists, respectively
            self.Som.append(stiffness_matrix)
            self.Com.append(compliance_matrix)

            # Calculate the 6x6 stress and strain rotation matrices
            angle = layer.angle
            c = np.cos(np.radians(angle))
            s = np.sin(np.radians(angle))

            Ts = np.array(
                [
                    [c**2, s**2, 0, 0, 0, 2 * s * c],
                    [s**2, c**2, 0, 0, 0, -2 * s * c],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, -s, 0],
                    [0, 0, 0, s, c, 0],
                    [-s * c, s * c, 0, 0, 0, c**2 - s**2],
                ]
            )

            Te = np.array(
                [
                    [c**2, s**2, 0, 0, 0, s * c],
                    [s**2, c**2, 0, 0, 0, -s * c],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, c, -s, 0],
                    [0, 0, 0, s, c, 0],
                    [-2 * s * c, 2 * s * c, 0, 0, 0, c**2 - s**2],
                ]
            )

            # Append the stress and strain rotation matrices and their inverses to the Tsig, Tsig_1, Teps, and Teps_1 lists
            self.Tsig.append(Ts)
            self.Tsig_1.append(np.linalg.inv(Ts))
            self.Teps.append(Te)
            self.Teps_1.append(np.linalg.inv(Te))

            # calculate the transformed compliance matrix for the current layer and appends it to the Cgm list:
            tmp_cgm = self.Tsig_1[-1] @ compliance_matrix @ Te

            self.Cgm.append(tmp_cgm)

    def __calculate_coefficients(self):
        # Get the layers from the tank
        layers: list[Layer] = self.tank.layers

        # Calculate the matrices needed for the coefficients
        self.__calculate_matrices()
        Cgm: list[np.ndarray] = self.Cgm

        # Loop through each layer to calculate the coefficients
        for i, layer in enumerate(layers):
            # Calculate the position of the current layer in the tank
            self.r_cl[i + 1] = self.r_cl[i] + layer.thickness

            # Get the compliance matrix from the previous step
            # Calculate the a1, a2, and b coefficients
            self.a1[i] = (Cgm[i][0][1] - Cgm[i][0][2]) / (Cgm[i][2][2] - Cgm[i][1][1])
            self.a2[i] = (Cgm[i][1][5] - 2 * Cgm[i][2][5]) / (
                4 * Cgm[i][2][2] - Cgm[i][1][1]
            )
            self.b[i] = np.sqrt(Cgm[i][1][1] / Cgm[i][2][2])

        # Create arrays to store the d, e, and a coefficients
        d = np.zeros((2 * self.number_of_layers + 2, self.number_of_layers))
        e = np.zeros((2 * self.number_of_layers + 2, self.number_of_layers))
        a = np.zeros((2 * self.number_of_layers + 2, 2))

        # Define boundary conditions
        d[0, 0] = (Cgm[0][1, 2] + self.b[0] * Cgm[0][2, 2]) * self.r_cl[0] ** (
            self.b[0] - 1
        )  # First boundary condition for d

        e[0, 0] = (Cgm[0][1, 2] - self.b[0] * Cgm[0][2, 2]) * self.r_cl[0] ** (
            -self.b[0] - 1
        )  # First boundary condition for e

        a[0, 0] = Cgm[0][0, 2] + self.a1[0] * (
            Cgm[0][1, 2] + Cgm[0][2, 2]
        )  # First boundary condition for a

        a[0, 1] = (
            Cgm[0][2, 5] + self.a2[0] * (Cgm[0][1, 2] + 2 * Cgm[0][2, 2])
        ) * self.r_cl[
            0
        ]  # Second boundary condition for a

        a[2 * self.number_of_layers - 1, 0] = Cgm[self.number_of_layers - 1][
            0, 2
        ] + self.a1[self.number_of_layers - 1] * (
            Cgm[self.number_of_layers - 1][1, 2] + Cgm[self.number_of_layers - 1][2, 2]
        )  # Last boundary condition for a

        a[2 * self.number_of_layers - 1, 1] = (
            Cgm[self.number_of_layers - 1][2, 5]
            + self.a2[self.number_of_layers - 1]
            * (
                Cgm[self.number_of_layers - 1][1, 2]
                + 2 * Cgm[self.number_of_layers - 1][2, 2]
            )
        ) * self.r_cl[
            self.number_of_layers - 1
        ]  # Second last boundary condition for a

        a[2 * self.number_of_layers, 0] = 0  # Boundary condition for a, bottom row
        a[2 * self.number_of_layers, 1] = 0  # Boundary condition for a, bottom row

        a[2 * self.number_of_layers + 1, 0] = 0  # Boundary condition for a, top row
        a[2 * self.number_of_layers + 1, 1] = 0  # Boundary condition for a, top row

        # Loop over each layer, except the first (already set in previous block)
        for i in range(1, self.number_of_layers):
            # Set diagonal elements of d and e matrices
            d[i, i] = -self.r_cl[i - 1] ** self.b[i]
            e[i, i] = -self.r_cl[i - 1] ** (-self.b[i])

            # Set elements of a matrix for the i-th row
            a[i, 0] = (self.a1[i - 1] - self.a1[i]) * self.r_cl[i - 1]
            a[i, 1] = (self.a2[i - 1] - self.a2[i]) * self.r_cl[i - 1] ** 2

            # Set elements of a matrix for the i-th row of the second half
            a[i + self.number_of_layers - 1, 0] = (
                (Cgm[i - 1][0, 2] - Cgm[i][0, 2])
                + self.a1[i - 1] * (Cgm[i - 1][1, 2] + Cgm[i - 1][2, 2])
                - self.a1[i] * (Cgm[i][1, 2] + Cgm[i][2, 2])
            )
            a[i + self.number_of_layers - 1, 1] = (
                (Cgm[i - 1][2, 5] - Cgm[i][2, 5])
                + self.a2[i - 1] * (Cgm[i - 1][1, 2] + 2 * Cgm[i - 1][2, 2])
                - self.a2[i] * (Cgm[i][1, 2] + 2 * Cgm[i][2, 2])
            ) * self.r_cl[i - 1]

            # Set off-diagonal elements of d and e matrices
            d[i, i - 1] = self.r_cl[i - 1] ** self.b[i - 1]
            e[i, i - 1] = self.r_cl[i - 1] ** (-self.b[i - 1])

            # Set elements of d and e matrices for the i-th row of the second half
            d[i + self.number_of_layers - 1, i] = -(
                Cgm[i][1, 2] + self.b[i] * Cgm[i][2, 2]
            ) * self.r_cl[i - 1] ** (self.b[i] - 1)
            e[i + self.number_of_layers - 1, i] = -(
                Cgm[i][1, 2] - self.b[i] * Cgm[i][2, 2]
            ) * self.r_cl[i - 1] ** (-self.b[i] - 1)

        for i in range(self.number_of_layers):
            d[i + self.number_of_layers - 1][i] = (
                Cgm[i][1][2] + self.b[i] * Cgm[i][2][2]
            ) * self.r_cl[i + 1] ** (self.b[i] - 1)
            e[i + self.number_of_layers - 1][i] = (
                Cgm[i][1][2] - self.b[i] * Cgm[i][2][2]
            ) * self.r_cl[i + 1] ** (-self.b[i] - 1)
            d[2 * self.number_of_layers][i] = (
                (Cgm[i][0][1] + self.b[i] * Cgm[i][0][2])
                / (1 + self.b[i])
                * (
                    self.r_cl[i + 1] ** (self.b[i] + 1)
                    - self.r_cl[i + 1 - 1] ** (self.b[i] + 1)
                )
            )
            e[2 * self.number_of_layers][i] = (
                (Cgm[i][0][1] - self.b[i] * Cgm[i][0][2])
                / (1 - self.b[i])
                * (
                    self.r_cl[i + 1] ** (-self.b[i] + 1)
                    - self.r_cl[i + 1 - 1] ** (-self.b[i] + 1)
                )
            )
            d[2 * self.number_of_layers + 1][i] = (
                (Cgm[i][1][5] + self.b[i] * Cgm[i][2][5])
                / (2 + self.b[i])
                * (
                    self.r_cl[i + 1] ** (self.b[i] + 2)
                    - self.r_cl[i + 1 - 1] ** (self.b[i] + 2)
                )
            )
            e[2 * self.number_of_layers + 1][i] = (
                (Cgm[i][1][5] - self.b[i] * Cgm[i][2][5])
                / (2 - self.b[i])
                * (
                    self.r_cl[i + 1] ** (-self.b[i] + 2)
                    - self.r_cl[i + 1 - 1] ** (-self.b[i] + 2)
                )
            )
            a[2 * self.number_of_layers][0] += (
                (Cgm[i][0][0] + self.a1[i] * (Cgm[i][0][1] + Cgm[i][0][2]))
                * (self.r_cl[i + 1] ** 2 - self.r_cl[i + 1 - 1] ** 2)
                / 2
            )
            a[2 * self.number_of_layers][1] += (
                (Cgm[i][0][5] + self.a2[i] * (Cgm[i][0][1] + 2 * Cgm[i][0][2]))
                * (self.r_cl[i + 1] ** 3 - self.r_cl[i + 1 - 1] ** 3)
                / 3
            )
            a[2 * self.number_of_layers + 1][0] += (
                (Cgm[i][0][5] + self.a1[i] * (Cgm[i][1][5] + Cgm[i][2][5]))
                * (self.r_cl[i + 1] ** 3 - self.r_cl[i + 1 - 1] ** 3)
                / 3
            )
            a[2 * self.number_of_layers + 1][1] += (
                (Cgm[i][5][5] + self.a2[i] * (Cgm[i][1][5] + 2 * Cgm[i][2][5]))
                * (self.r_cl[i + 1] ** 4 - self.r_cl[i + 1 - 1] ** 4)
                / 4
            )

        # set d, e, a
        self.d = d
        self.e = e
        self.a = a

    def calculate_constantes(self):
        self.__calculate_coefficients()
        P = self.tank.burst_test_pressure
        VCL = np.zeros(2 * self.number_of_layers + 2)
        VCL[0] = -P
        VCL[2 * self.number_of_layers + 1] = self.r_cl[0] ** 2 * P / 2

        # create the matrice
        # [d, e ,a]

        MatSYS = np.concatenate((self.d, self.e, self.a), axis=1)
        MatSYS_1 = np.linalg.inv(MatSYS)

        Vconst = np.zeros(2 * self.number_of_layers + 1)

        for i in range(2 * self.number_of_layers + 1):
            Vconst[i] = np.dot(MatSYS_1[i], VCL)

        computation_radius = np.zeros(
            self.number_of_layers * self.number_of_points_by_layer
        )

        for i in range(self.number_of_layers):
            current_layer: Layer = self.tank.layers[i]
            radius_step = current_layer.thickness / (self.number_of_points_by_layer - 1)

            computation_radius[
                i
                * self.number_of_points_by_layer : (i + 1)
                * self.number_of_points_by_layer
            ] = (self.r_cl[i] + np.arange(self.number_of_points_by_layer) * radius_step)

        # also known as the thickness of the composite layer
        normalized_radius_denominator = self.r_cl[-1] - self.r_cl[0]

        normalized_radius = (
            computation_radius - self.r_cl[0]
        ) / normalized_radius_denominator

        return normalized_radius
