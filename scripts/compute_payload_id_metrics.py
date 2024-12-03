import numpy as np


def MassErrorMetric(estimated_mass, true_mass):
    return 100 * (abs(estimated_mass - true_mass) / true_mass)


# Uses the extends of the bounding-box of the object to put the error in perspective
def CentreOfMassErrorMetric(estimated_com_c, true_com_c, true_boundingbox):
    error_vector = np.array([0, 0, 0])
    for i in range(0, 3):
        try:
            error_vector[i] = 100 * (
                abs(estimated_com_c[i] - true_com_c[i]) / abs(true_boundingbox[i])
            )
        except OverflowError:
            error_vector[i] = np.Inf
    return error_vector


# Computes the error metric of the inertia tensor
def InertiaErrorMetric(
    estimated_Inertia_tensor, true_Inertia_tensor, true_mass, true_boundingbox
):
    a = abs(true_boundingbox)
    error_matrix = np.zeros((3, 3))
    for i in range(0, 3):
        for j in range(0, 3):
            if i == j:
                delta = 1
            else:
                delta = 0
            numer = estimated_Inertia_tensor[i, j] - true_Inertia_tensor[i, j]
            denom = (true_mass / 12) * (
                delta * (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) - a[i] * a[j]
            )
            error_matrix[i, j] = 100 * abs(numer / denom)
    return error_matrix


if __name__ == "__main__":
    bounding_box = np.array([0.2, 0.315, 0.05])

    gt_mass = 1.373
    gt_com = np.array([0.0000000000, -0.0274217369, 0.0250000000])
    gt_inertia = np.array(
        [
            [0.0150943475, 0.0000000000, 0.0000000000],
            [0.0000000000, 0.0032957768, 0.0009468168],
            [0.0000000000, 0.0009468168, 0.0162788582],
        ]
    )

    measured_mass = 1.3787098987764943
    measured_com = np.array([0.00383134, -0.02965649, 0.00636925])
    measured_inertia = np.array(
        [
            [0.03866183, 0.0029429, 0.00836289],
            [0.0029429, 0.03585378, -0.01091643],
            [0.00836289, -0.01091643, 0.00703691],
        ]
    )

    mass_error = MassErrorMetric(measured_mass, gt_mass)
    com_error = CentreOfMassErrorMetric(measured_com, gt_com, bounding_box)
    inertia_error = InertiaErrorMetric(
        measured_inertia, gt_inertia, gt_mass, bounding_box
    )

    print("Mass error: ", mass_error)
    print("Centre of mass error: ", np.mean(com_error))
    print("Inertia error: ", np.mean(inertia_error[np.triu_indices(3)]))
