from src.models.tank import Tank, Layer
import numpy as np
import pandas as pd
import uuid


class Computation:
    def __init__(self, tank: Tank, number_of_points_by_layer: int):
        self.tank: Tank = tank
        self.normalized_length = 1
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

        self.Sigg: np.array = np.array([])  # used as a flag in save_data_to_excel

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
            # can be improved if the same material is used
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
            # can be improved if the same angle is already calculated
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
        ) * self.r_cl[self.number_of_layers]
        """
            You probably need to check self.number_of_layers - 1 
        """
        # Second last boundary condition for a

        a[2 * self.number_of_layers, 0] = 0  # Boundary condition for a, bottom row
        a[2 * self.number_of_layers, 1] = 0  # Boundary condition for a, bottom row

        a[2 * self.number_of_layers + 1, 0] = 0  # Boundary condition for a, top row
        a[2 * self.number_of_layers + 1, 1] = 0  # Boundary condition for a, top row

        # Loop over each layer, except the first (already set in previous block)
        for i in range(1, self.number_of_layers):
            # Set diagonal elements of d and e matrices
            d[i, i] = -self.r_cl[i] ** (self.b[i])
            e[i, i] = -self.r_cl[i] ** (-self.b[i])

            # Set elements of a matrix for the i-th row
            a[i, 0] = (self.a1[i - 1] - self.a1[i]) * self.r_cl[i]
            a[i, 1] = (self.a2[i - 1] - self.a2[i]) * (self.r_cl[i] ** 2)

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
            ) * self.r_cl[i]

            # Set off-diagonal elements of d and e matrices
            d[i, i - 1] = self.r_cl[i] ** self.b[i - 1]
            e[i, i - 1] = self.r_cl[i] ** (-self.b[i - 1])

            # Set elements of d and e matrices for the i-th row of the second half
            d[i + self.number_of_layers - 1, i] = -(
                Cgm[i][1, 2] + self.b[i] * Cgm[i][2, 2]
            ) * self.r_cl[i] ** (self.b[i] - 1)
            e[i + self.number_of_layers - 1, i] = -(
                Cgm[i][1, 2] - self.b[i] * Cgm[i][2, 2]
            ) * self.r_cl[i] ** (-self.b[i] - 1)

        for i in range(self.number_of_layers):
            d[i + self.number_of_layers][i] = (
                Cgm[i][1][2] + self.b[i] * Cgm[i][2][2]
            ) * self.r_cl[i + 1] ** (self.b[i] - 1)
            e[i + self.number_of_layers][i] = (
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
        VCL[2 * self.number_of_layers] = self.r_cl[0] ** 2 * P / 2

        # create the matrice
        # [d, e ,a]

        MatSYS = np.concatenate((self.d, self.e, self.a), axis=1)
        MatSYS_1 = np.linalg.inv(MatSYS)

        Vconst = np.zeros(2 * self.number_of_layers + 2)

        for i in range(2 * self.number_of_layers + 2):
            Vconst[i] = np.dot(MatSYS_1[i], VCL)

        self.VCL = VCL
        self.Vconst = Vconst

        return Vconst

    def calculate_displacement(self):
        Vconst = self.calculate_constantes()

        computation_radius = np.zeros(
            self.number_of_layers * self.number_of_points_by_layer
        )

        Ur = np.zeros(self.number_of_layers * self.number_of_points_by_layer)
        Uq = np.zeros(self.number_of_layers * self.number_of_points_by_layer)
        Uz = np.zeros(self.number_of_layers * self.number_of_points_by_layer)

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

        for i in range(self.number_of_layers):
            start_index = i * self.number_of_points_by_layer
            end_index = (i + 1) * self.number_of_points_by_layer
            r = computation_radius[start_index:end_index]
            Ur[start_index:end_index] = (
                Vconst[i] * (r ** self.b[i])
                + Vconst[self.number_of_layers + i] * (r ** (-self.b[i]))
                + self.a1[i] * Vconst[2 * self.number_of_layers] * r
                + self.a2[i] * Vconst[2 * self.number_of_layers + 1] * (r**2)
            )
            #  a1(i) * Vconst(2 * NBC + 1) * r(k) + a2(i) * Vconst(2 * NBC + 2) * r(k) ^ 2
            Uq[start_index:end_index] = (
                Vconst[2 * self.number_of_layers + 1] * r * self.normalized_length / 2
            )
            Uz[start_index:end_index] = (
                -Vconst[2 * self.number_of_layers] * self.normalized_length / 2
            )
        self.computation_radius = computation_radius
        self.normalized_radius = normalized_radius
        self.Ur = Ur
        self.Uq = Uq
        self.Uz = Uz

    def calculate_deformation_and_constraints(self):
        self.calculate_displacement()

        self.Epsg: np.ndarray = np.zeros(
            (6, self.number_of_layers * self.number_of_points_by_layer)
        )
        self.Sigg: np.array = np.zeros(
            (self.number_of_layers * self.number_of_points_by_layer, 6)
        )
        self.Epso: np.array = np.zeros(
            (self.number_of_layers * self.number_of_points_by_layer, 6)
        )
        self.Sigo: np.array = np.zeros(
            (self.number_of_layers * self.number_of_points_by_layer, 6)
        )

        for i in range(self.number_of_layers):
            start, end = (
                i * self.number_of_points_by_layer,
                (i + 1) * self.number_of_points_by_layer,
            )
            self.Epsg[0, start:end] = self.Vconst[2 * self.number_of_layers]
            self.Epsg[1, start:end] = (
                self.Vconst[i]
                * np.power(self.computation_radius[start:end], (self.b[i] - 1))
                + self.Vconst[self.number_of_layers + i]
                * np.power(self.computation_radius[start:end], (-self.b[i] - 1))
                + self.a1[i] * self.Vconst[2 * self.number_of_layers]
                + self.a2[i]
                * self.Vconst[2 * self.number_of_layers + 1]
                * self.computation_radius[start:end]
            )

            self.Epsg[2, start:end] = (
                self.b[i]
                * self.Vconst[i]
                * np.power(self.computation_radius[start:end], (self.b[i] - 1))
                - self.b[i]
                * self.Vconst[self.number_of_layers + i]
                * np.power(self.computation_radius[start:end], (-self.b[i] - 1))
                + self.a1[i] * self.Vconst[2 * self.number_of_layers]
                + 2
                * self.a2[i]
                * self.Vconst[2 * self.number_of_layers + 1]
                * self.computation_radius[start:end]
            )

            self.Epsg[5, start:end] = (
                self.Vconst[2 * self.number_of_layers + 1]
                * self.computation_radius[start:end]
            )

            self.Sigg[start:end, :] = (self.Cgm[i] @ self.Epsg[:, start:end]).T
            self.Epso[start:end, :] = (self.Teps[i] @ self.Epsg[:, start:end]).T
            self.Sigo[start:end, :] = (self.Tsig[i] @ self.Sigg[start:end, :].T).T

    def __calculate_F(self, Xt, Xc, Yt, Yc, X12):
        """
        Calculates the Tsai-Wu coefficients F11, F22, F12, F23, F13, and F33.

        Parameters:
        -----------
        Xt: float
            Tensile strength in the fiber direction
        Xc: float
            Compressive strength in the fiber direction
        Yt: float
            Tensile strength in the transverse direction
        Yc: float
            Compressive strength in the transverse direction
        S: float
            Shear strength

        Returns:
        --------
        F1, F2, F11, F22, F66, F12: np.array
            Tsai-Wu coefficients array
        """

        F1 = 1 / Xt - 1 / Xc
        F2 = 1 / Yt - 1 / Yc

        F11 = 1 / (Xt * Xc)
        F22 = 1 / (Yt * Yc)
        F66 = 1 / (X12**2)
        F12 = -np.sqrt(F11 * F22)

        return np.array([F1, F2, F11, F22, F66, F12])

    def tsai_wu_failure_criteria(self):
        # Calculate the Tsai-Wu coefficients
        tsai_wu: np.array = np.zeros(
            (self.number_of_layers * self.number_of_points_by_layer)
        )
        F_material = dict()
        layers: list[Layer] = self.tank.layers

        for i, layer in enumerate(layers):
            material_name = layer.name

            if material_name not in F_material.keys():
                Xt, Xc, Yt, Yc, X12 = layer.material.get_constants()
                F_material[material_name] = self.__calculate_F(Xt, Xc, Yt, Yc, X12)

            for j in range(self.number_of_points_by_layer):
                index = i * self.number_of_points_by_layer + j

                s1, s2, s12 = self.Sigo[index][[0, 1, -1]]
                S = np.array([s1, s2, s1**2, s2**2, s12**2, s1 * s2])
                F = F_material[material_name]
                tsai_wu[index] = np.dot(S, F)

        return tsai_wu

    def save_data_to_excel(self, path: str = "computation_out/"):
        if len(self.Sigg) != 0:
            self.calculate_deformation_and_constraints()

        data = pd.DataFrame()
        data["r"] = self.computation_radius
        data["R"] = self.normalized_radius
        data["Ur"] = self.Ur
        data["Uq"] = self.Uq
        data["Uz"] = self.Uz

        data["eps1"] = self.Epsg.T[:, 0]
        data["eps2"] = self.Epsg.T[:, 1]
        data["eps3"] = self.Epsg.T[:, 2]
        data["eps4"] = self.Epsg.T[:, 3]
        data["eps5"] = self.Epsg.T[:, 4]
        data["eps6"] = self.Epsg.T[:, 5]

        data["epsx"] = self.Epso[:, 0]
        data["epsy"] = self.Epso[:, 1]
        data["epsz"] = self.Epso[:, 2]
        data["epsyz"] = self.Epso[:, 3]
        data["epsxz"] = self.Epso[:, 4]
        data["epsxy"] = self.Epso[:, 5]

        data["sig1"] = self.Sigg[:, 0]
        data["sig2"] = self.Sigg[:, 1]
        data["sig3"] = self.Sigg[:, 2]
        data["sig4"] = self.Sigg[:, 3]
        data["sig5"] = self.Sigg[:, 4]
        data["sig6"] = self.Sigg[:, 5]

        data["sigx"] = self.Sigo[:, 0]
        data["sigy"] = self.Sigo[:, 1]
        data["sigz"] = self.Sigo[:, 2]
        data["sigyz"] = self.Sigo[:, 3]
        data["sigxz"] = self.Sigo[:, 4]
        data["sigxy"] = self.Sigo[:, 5]

        unique_filename = "computation_data" + str(uuid.uuid4()) + ".xlsx"
        if path != None:
            unique_filename = path + unique_filename
        data.to_excel(unique_filename)
        print("Saved data in", unique_filename)
