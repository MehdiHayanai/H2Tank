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

    def calculate_matrices(self):
        layers: list[Layer] = self.tank.layers

        for layer in layers:
            # stfifness and compliance matrices
            E1, E2, E3 = layer.material.young_modulus
            G23, G13, G12 = layer.material.shear_modulus
            v23, v13, v12 = layer.material.poisson_ratios

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

            compliance_matrix = np.linalg.inv(stiffness_matrix)
            self.Som.append(stiffness_matrix)
            self.Com.append(compliance_matrix)

            # Rotation matrices

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

            self.Tsig.append(Ts)
            self.Tsig_1.append(np.linalg.inv(Ts))

            self.Teps.append(Te)
            self.Teps_1.append(np.linalg.inv(Te))

            tmp_cgm = self.Tsig_1[-1] @ compliance_matrix @ Te

            self.Cgm.append(tmp_cgm)

    def calculate_coefficients(self):
        # TO DO This section must be optimized
        layers: list[Layer] = self.tank.layers
        self.calculate_matrices()
        for i, layer in enumerate(layers):
            self.r_cl[i + 1] = self.r_cl[i] + layer.thickness
            Cgm = self.Cgm
            self.a1[i] = (Cgm[i][0][1] - Cgm[i][0][2]) / (Cgm[i][2][2] - Cgm[i][1][1])
            self.a2[i] = (Cgm[i][1][5] - 2 * Cgm[i][2][5]) / (
                4 * Cgm[i][2][2] - Cgm[i][1][1]
            )
            self.b[i] = np.sqrt(Cgm[i][1][1] / Cgm[i][2][2])

        d = np.zeros((2 * self.number_of_layers + 2, self.number_of_layers))
        e = np.zeros((2 * self.number_of_layers + 2, self.number_of_layers))
        a = np.zeros((2 * self.number_of_layers + 2, 2))

        d[0, 0] = (Cgm[0][1, 2] + self.b[0] * Cgm[0][2, 2]) * self.r_cl[0] ** (
            self.b[0] - 1
        )
        e[0, 0] = (Cgm[0][1, 2] - self.b[0] * Cgm[0][2, 2]) * self.r_cl[0] ** (
            -self.b[0] - 1
        )

        a[0, 0] = Cgm[0][0, 2] + self.a1[0] * (Cgm[0][1, 2] + Cgm[0][2, 2])

        a[0, 1] = (
            Cgm[0][2, 5] + self.a2[0] * (Cgm[0][1, 2] + 2 * Cgm[0][2, 2])
        ) * self.r_cl[0]

        a[2 * self.number_of_layers - 1, 0] = Cgm[self.number_of_layers - 1][
            0, 2
        ] + self.a1[self.number_of_layers - 1] * (
            Cgm[self.number_of_layers - 1][1, 2] + Cgm[self.number_of_layers - 1][2, 2]
        )
        a[2 * self.number_of_layers - 1, 1] = (
            Cgm[self.number_of_layers - 1][2, 5]
            + self.a2[self.number_of_layers - 1]
            * (
                Cgm[self.number_of_layers - 1][1, 2]
                + 2 * Cgm[self.number_of_layers - 1][2, 2]
            )
        ) * self.r_cl[self.number_of_layers - 1]
        a[2 * self.number_of_layers, 0] = 0
        a[2 * self.number_of_layers, 1] = 0
        a[2 * self.number_of_layers + 1, 0] = 0
        a[2 * self.number_of_layers + 1, 1] = 0

        for i in range(1, self.number_of_layers):
            d[i, i] = -self.r_cl[i - 1] ** self.b[i]
            e[i, i] = -self.r_cl[i - 1] ** (-self.b[i])
            a[i, 0] = (self.a1[i - 1] - self.a1[i]) * self.r_cl[i - 1]
            a[i, 1] = (self.a2[i - 1] - self.a2[i]) * self.r_cl[i - 1] ** 2
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
            d[i, i - 1] = self.r_cl[i - 1] ** self.b[i - 1]
            e[i, i - 1] = self.r_cl[i - 1] ** (-self.b[i - 1])
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
