"""
1. Estimate mass and CoM (much easier than inertia)
2. Solve an over-determined least-squared with point masses and mass, com constraints
"""

import numpy as np
import trimesh

# Load mesh
mesh = trimesh.load("models/test_object.obj")
mass = 1.381118658237185  # Total mass
com_given = np.array(
    [0.0000000000, -0.0274217369, 0.0250000000]
)  # Given center of mass

# Get triangle areas and centers
areas = mesh.area_faces
triangle_coms = mesh.triangles_center

# Set up the constraint matrix
A = np.vstack(
    [
        areas,  # Total mass constraint
        areas * triangle_coms[:, 0],  # X component of COM
        areas * triangle_coms[:, 1],  # Y component of COM
        areas * triangle_coms[:, 2],  # Z component of COM
    ]
)
b = np.array([mass, mass * com_given[0], mass * com_given[1], mass * com_given[2]])

# Solve for densities (overdetermined least squares)
rho, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# Compute inertia tensor
inertia_tensor = np.zeros((3, 3))
for i, face in enumerate(mesh.faces):
    vertices = mesh.vertices[face]
    triangle_mass = rho[i] * areas[i]  # Triangle mass
    triangle_com = triangle_coms[i]  # Triangle COM
    d = triangle_com - com_given  # Vector to global COM

    # Parallel axis theorem for triangle contribution
    d_outer = np.outer(d, d)
    inertia_tensor += triangle_mass * (np.dot(d, d) * np.eye(3) - d_outer)

print("Mass (Given):", mass)
print("Center of Mass (Given):", com_given)
print("Estimated Inertia Tensor (about CoM):\n", inertia_tensor)
